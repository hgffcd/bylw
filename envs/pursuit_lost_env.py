# -*- coding: utf-8 -*-
"""短时失观多无人机协同追踪围捕环境（毕业论文核心环境）.

特性:
- N 架同质追捕无人机, unicycle 运动学: 状态 [x, y, psi, v], 动作 [a, omega] (归一化到 [-1,1]);
- 单机动目标: 人工势场(APF)逃逸策略, 速度比可配置;
- 静态圆形障碍 + 动态圆形障碍(匀速反弹), 参与遮挡判定;
- 可见性 = 距离 + 扇形视场 + 视线遮挡 三条件; 团队级失观计数 t_lost;
- 失观处理模式 obs_mode: oracle(真值,调试用)/zero(清零+标志)/last(最后观测)/belief(外部信念注入);
- 围捕成功 = 距离闭合(>=min_capture_agents 机进入捕获半径) 且 相邻围捕角最大缺口 <= max_gap;
- 团队奖励: 捕获 + 势函数差分接近 + 围捕角均匀 + 重捕获 + 碰撞/时间惩罚 (PBRS形式);
- info 中输出失观/安全细粒度统计, 供论文指标体系使用.

对应开题报告: 第5.1节建模, 式(5-1)~(5-7); 奖励对应式(5-19)~(5-22).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.geometry import (
    segment_circle_intersect,
    wrap_angle,
    sorted_gap_max,
)


@dataclass
class EnvConfig:
    # 场地与回合
    arena_size: float = 100.0          # 正方形边长 [m]
    dt: float = 0.1                    # 步长 [s]
    max_steps: int = 500
    # 追捕者
    n_pursuers: int = 3
    v_max: float = 8.0                 # 最大速度 [m/s]
    a_max: float = 4.0                 # 最大加速度 [m/s^2]
    w_max: float = 1.5                 # 最大角速度 [rad/s]
    pursuer_radius: float = 0.5
    # 目标
    speed_ratio: float = 0.9           # 目标最大速度 / 追捕最大速度
    target_radius: float = 0.5
    evader_noise: float = 0.15         # APF 逃逸随机扰动强度
    evader_type: str = "apf"           # apf | mode_switch | reactive_mode_switch
    evader_mode_dwell: tuple = (2.0, 4.0)  # mode_switch: 模式停留时长范围 [s]
    evader_turn_rate: float = 0.3      # mode_switch: CT 模式转弯率 [rad/s]
    evader_pursuer_repulsion_gain: float = 1.0  # reactive_mode_switch: 追捕者APF增益
    evader_pursuer_reaction_range: float = 30.0  # reactive_mode_switch: 反应距离[m]
    # 感知
    sense_range: float = 25.0          # 探测距离 R_s
    fov_half: float = math.radians(60) # 视场半角 phi
    lost_cap_steps: int = 100          # 短时失观上限 (10 s)
    target_position_noise_std_m: float = 0.0  # J3: 可见目标位置测量噪声, 默认关闭
    visual_occlusion: bool = True       # T2/T4: 障碍是否遮挡视线; 不影响碰撞几何
    visibility_diagnostics: bool = False  # T2: 记录range/fov/occlusion原因, 默认关闭
    require_initial_visibility: bool = False  # T2: 从真实已探测状态开始，避免把初始搜索误计为失观
    # P4 causal evaluation only: draw the tracked-start geometry with one
    # frozen sensing reference, then apply the runtime sensing intervention.
    # None preserves every existing training/evaluation reset exactly.
    initial_visibility_reference_sense_range: Optional[float] = None
    initial_visibility_reference_visual_occlusion: Optional[bool] = None
    # Opt-in team communication: once any pursuer has a direct detection,
    # broadcast the current target position/velocity and an available flag to
    # every pursuer.  Local visibility and its range/FOV/occlusion causes stay
    # unchanged in telemetry.  False preserves all frozen experiment paths.
    team_target_sharing: bool = False
    # Opt-in causal intervention: suppress all target measurements in the
    # half-open interval [start, start + duration).  Defaults preserve the
    # frozen path exactly.
    measurement_blackout_start_step: Optional[int] = None
    measurement_blackout_duration_steps: int = 0
    # 障碍
    n_static_obs: int = 8
    n_dynamic_obs: int = 2
    obs_radius_range: tuple = (2.0, 6.0)
    dyn_obs_speed: float = 2.0
    k_nearest_obs: int = 4             # 观测中的最近障碍数
    obstacle_layout: str = "random"    # random | corridor | cluster | narrow_passage
    # 围捕判定
    capture_radius: float = 5.0        # rho_c
    min_capture_agents: int = 3
    max_gap: float = math.pi           # 相邻围捕角最大缺口 theta_max
    # 安全
    d_safe_agent: float = 2.5          # 机间基础安全距离(指标统计用)
    # 奖励权重
    r_capture: float = 100.0
    k_dist: float = 1.0                # 势函数差分系数
    k_ang: float = 0.5                 # 围捕角均匀性(仅在接近段生效)
    k_rec: float = 5.0                 # 重捕获奖励
    k_col: float = 20.0                # 碰撞惩罚
    k_time: float = 0.02               # 每步时间惩罚
    # J11 opt-in reward terms.  Defaults are zero so all frozen E1--J10
    # trajectories and returns remain unchanged.
    k_visible: float = 0.0             # 团队保持至少一机可见的逐步奖励
    k_lost: float = 0.0                # 随连续失观时长线性增长的逐步惩罚
    r_long_lost: float = 0.0           # 触及long-loss终止边界的终端惩罚
    r_timeout: float = 0.0             # 未完成围捕而超时的终端惩罚
    gamma_pbrs: float = 0.99           # PBRS 折扣(与训练折扣一致)
    reward_target_reference: str = "observation"  # observation | truth
    # 奖励期望围捕环半径；None时沿用capture_radius，保证历史配置路径不变。
    reward_target_radius: Optional[float] = None
    collision_terminates: bool = True
    # 观测/失观模式
    obs_mode: str = "zero"             # oracle | zero | last | belief
    # 课程接口(失观强度轴/难度轴的关键参数都在上方, 由外部配置覆盖)
    seed: Optional[int] = None


class PursuitLostEnv:
    """numpy 实现, 无 torch 依赖, 便于向量化与单元测试."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        if self.cfg.obstacle_layout not in {
                "random", "corridor", "cluster", "narrow_passage"}:
            raise ValueError(
                "obstacle_layout must be random/corridor/cluster/narrow_passage")
        if self.cfg.target_position_noise_std_m < 0:
            raise ValueError("target_position_noise_std_m must be non-negative")
        if self.cfg.measurement_blackout_duration_steps < 0:
            raise ValueError(
                "measurement_blackout_duration_steps must be non-negative")
        if (
            self.cfg.measurement_blackout_start_step is not None
            and self.cfg.measurement_blackout_start_step < 0
        ):
            raise ValueError(
                "measurement_blackout_start_step must be non-negative or None")
        if self.cfg.reward_target_reference not in {"observation", "truth"}:
            raise ValueError(
                "reward_target_reference must be observation or truth")
        if (
            self.cfg.reward_target_radius is not None
            and self.cfg.reward_target_radius <= 0
        ):
            raise ValueError("reward_target_radius must be positive or None")
        if self.cfg.evader_type not in {
                "apf", "mode_switch", "reactive_mode_switch"}:
            raise ValueError(
                "evader_type must be apf/mode_switch/reactive_mode_switch")
        if self.cfg.evader_pursuer_repulsion_gain < 0:
            raise ValueError("evader_pursuer_repulsion_gain must be non-negative")
        if self.cfg.evader_pursuer_reaction_range <= 0:
            raise ValueError("evader_pursuer_reaction_range must be positive")
        if (
            self.cfg.require_initial_visibility
            and self.cfg.sense_range <= self.cfg.capture_radius
        ):
            raise ValueError(
                "require_initial_visibility needs sense_range > capture_radius"
            )
        if (
            self.cfg.initial_visibility_reference_sense_range is not None
            and self.cfg.initial_visibility_reference_sense_range
            <= self.cfg.capture_radius
        ):
            raise ValueError(
                "initial visibility reference needs range > capture_radius"
            )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.obs_rng = np.random.default_rng(self._observation_seed(self.cfg.seed))
        self.n = self.cfg.n_pursuers
        self._build_spaces()
        self.reset()

    @staticmethod
    def _observation_seed(seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return None
        return (int(seed) ^ 0x5DEECE66D) % (2 ** 32)

    # ---------------- 空间定义 ----------------
    def _build_spaces(self):
        c = self.cfg
        # 单机观测 = 自身(5) + 目标块(6) + 队友(4*(n-1)) + 障碍(5*k)
        self.obs_dim = 5 + 6 + 4 * (self.n - 1) + 5 * c.k_nearest_obs
        # 全局状态 = n*(x,y,cos,sin,v) + 目标(x,y,vx,vy) + t_lost_norm + 障碍(k*3)
        self.state_dim = 5 * self.n + 4 + 1 + 3 * c.k_nearest_obs
        self.act_dim = 2

    # ---------------- 重置 ----------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.obs_rng = np.random.default_rng(self._observation_seed(seed))
        c = self.cfg
        L = c.arena_size
        self.t = 0
        # 障碍: 中部区域随机, 不与出生点重叠
        self.obs_pos, self.obs_r, self.obs_vel = self._sample_obstacles()
        # 追捕者: 左侧带状区域; 目标: 右侧。T2可选从“已探测”状态开始，
        # 使 t_lost 只表示真实可见后的中断，而不是起点到目标的长距离搜索。
        if c.require_initial_visibility:
            self.t_pos = self._sample_free(
                np.array([0.85 * L, 0.5 * L]), 6.0)
            self.p_pos = self._sample_initially_visible_pursuers()
            delta = self.t_pos[None, :] - self.p_pos
            self.p_psi = np.arctan2(delta[:, 1], delta[:, 0])
        else:
            # Preserve the historical RNG call order and reset geometry exactly.
            self.p_pos = np.stack([
                self._sample_free(np.array([
                    0.1 * L,
                    (0.15 + 0.7 * i / max(self.n - 1, 1)) * L,
                ]), 5.0)
                for i in range(self.n)
            ])
            self.p_psi = self.rng.uniform(-math.pi, math.pi, self.n)
            self.t_pos = self._sample_free(
                np.array([0.85 * L, 0.5 * L]), 6.0)
        self.p_v = np.zeros(self.n)
        self.t_vel = np.zeros(2)
        self.target_measurement_pos = self.t_pos.copy()
        # 失观状态
        self.t_lost = 0                      # 团队级连续失观步数
        self.last_seen_rel = np.zeros((self.n, 4))  # 各机最后一次可见时的相对量
        self.last_seen_pos = self.t_pos.copy()      # 团队最后观测的目标绝对位置
        self.visible_each = np.zeros(self.n, bool)
        self.visibility_reasons_each = [tuple() for _ in range(self.n)]
        self.team_visibility_status = "visible"
        self.prev_visible_team = True
        self.belief = None                   # 外部信念注入接口: dict(mu, sigma, conf)
        # 统计
        self.stats = dict(
            lost_events=0,
            max_lost=0,
            recaptures=0,
            min_agent_dist=np.inf,
            min_obs_dist=np.inf,
            min_static_obs_dist=np.inf,
            min_dynamic_obs_dist=np.inf,
            collisions=0,
            collision_agent_steps=0,
            collision_static_steps=0,
            collision_dynamic_steps=0,
            lost_steps=0,
        )
        if c.visibility_diagnostics:
            self.stats.update(
                agent_visibility_checks=0,
                agent_visible_checks=0,
                agent_loss_checks_range=0,
                agent_loss_checks_fov=0,
                agent_loss_checks_occlusion=0,
                agent_loss_checks_measurement_blackout=0,
                loss_steps_pure_range=0,
                loss_steps_pure_fov=0,
                loss_steps_pure_occlusion=0,
                loss_steps_mixed=0,
                loss_steps_measurement_blackout=0,
                loss_events_pure_range=0,
                loss_events_pure_fov=0,
                loss_events_pure_occlusion=0,
                loss_events_mixed=0,
                loss_events_measurement_blackout=0,
            )
        self._last_collision_details = dict(
            agent=False, static=False, dynamic=False)
        self._phi_prev = self._potential()
        self._update_visibility()
        self._reward_ref_prev = self._target_ref_source()
        if c.require_initial_visibility and not bool(self.visible_each.any()):
            raise RuntimeError(
                "initial-visibility sampler produced an invisible reset"
            )
        return self._get_obs(), self._get_state()

    def _sample_initially_visible_pursuers(self) -> np.ndarray:
        """Place pursuers on unobstructed, non-capturing bearings of the target.

        This branch is opt-in so all frozen journal configurations retain their
        original reset distribution.  Candidate bearings are deterministic for
        a seed and use the same line-of-sight geometry as runtime visibility.
        """
        c = self.cfg
        L = c.arena_size
        reference_range = (
            c.sense_range
            if c.initial_visibility_reference_sense_range is None
            else c.initial_visibility_reference_sense_range
        )
        reference_occlusion = (
            c.visual_occlusion
            if c.initial_visibility_reference_visual_occlusion is None
            else c.initial_visibility_reference_visual_occlusion
        )
        outer = min(0.8 * reference_range, 0.25 * L)
        lower = c.capture_radius + max(2.0, 2.0 * c.pursuer_radius)
        radii = [
            max(lower, outer),
            max(lower, min(0.7 * reference_range, 0.22 * L)),
            max(lower, min(0.6 * reference_range, 0.18 * L)),
        ]
        # Start from the target's left side (the historical approach direction),
        # then cover the full circle to handle random obstacle layouts.
        phase = float(self.rng.uniform(-math.pi / 36.0, math.pi / 36.0))
        angles = math.pi + phase + np.linspace(
            -math.pi, math.pi, 144, endpoint=False)
        candidates = []
        for radius in radii:
            if not (c.capture_radius < radius <= reference_range):
                continue
            for angle in angles:
                point = self.t_pos + radius * np.array([
                    math.cos(angle), math.sin(angle)])
                if np.any(point < 1.0) or np.any(point > L - 1.0):
                    continue
                if any(
                    np.linalg.norm(point - center)
                    <= obstacle_radius + c.pursuer_radius + 0.5
                    for center, obstacle_radius in zip(
                        self.obs_pos, self.obs_r)
                ):
                    continue
                if reference_occlusion and any(
                    segment_circle_intersect(
                        point, self.t_pos, center, obstacle_radius)
                    for center, obstacle_radius in zip(
                        self.obs_pos, self.obs_r)
                ):
                    continue
                candidates.append(point)

        selected = []
        min_separation = max(c.d_safe_agent, 2.0 * c.pursuer_radius + 0.5)
        while candidates and len(selected) < self.n:
            if not selected:
                # Prefer the historical left-to-right approach direction.
                scores = [point[0] for point in candidates]
            else:
                # Farthest-point selection avoids initially clustered pursuers.
                scores = [
                    min(np.linalg.norm(point - other) for other in selected)
                    for point in candidates
                ]
            index = int(np.argmin(scores) if not selected else np.argmax(scores))
            point = candidates.pop(index)
            if selected and min(
                    np.linalg.norm(point - other) for other in selected
            ) < min_separation:
                continue
            selected.append(point)
        if len(selected) != self.n:
            raise RuntimeError(
                f"could not place {self.n} initially visible pursuers"
            )
        return np.asarray(selected, dtype=float)

    def _sample_obstacles(self):
        c = self.cfg
        if c.obstacle_layout != "random":
            return self._sample_structured_obstacles(c.obstacle_layout)
        L = c.arena_size
        pos, rad = [], []
        trials = 0
        while len(pos) < c.n_static_obs + c.n_dynamic_obs and trials < 1000:
            trials += 1
            p = self.rng.uniform(0.2 * L, 0.8 * L, 2)
            r = self.rng.uniform(*c.obs_radius_range)
            if all(np.linalg.norm(p - q) > r + rr + 2.0 for q, rr in zip(pos, rad)):
                pos.append(p)
                rad.append(r)
        pos = np.array(pos) if pos else np.zeros((0, 2))
        rad = np.array(rad) if rad else np.zeros(0)
        vel = np.zeros_like(pos)
        if c.n_dynamic_obs > 0 and len(pos) >= c.n_dynamic_obs:
            ang = self.rng.uniform(-math.pi, math.pi, c.n_dynamic_obs)
            vel[-c.n_dynamic_obs:] = c.dyn_obs_speed * np.stack([np.cos(ang), np.sin(ang)], -1)
        return pos, rad, vel

    def _sample_structured_obstacles(self, layout: str):
        """J3 fixed-topology OOD layouts; dynamic obstacles remain seed-random."""
        c = self.cfg
        L = c.arena_size
        templates = {
            "corridor": [
                (0.35, 0.35), (0.45, 0.35), (0.55, 0.35), (0.65, 0.35),
                (0.35, 0.65), (0.45, 0.65), (0.55, 0.65), (0.65, 0.65),
            ],
            "cluster": [
                (0.38, 0.35), (0.45, 0.28), (0.52, 0.35), (0.45, 0.42),
                (0.55, 0.65), (0.62, 0.58), (0.69, 0.65), (0.62, 0.72),
            ],
            "narrow_passage": [
                (0.50, 0.20), (0.50, 0.30), (0.50, 0.40),
                (0.50, 0.60), (0.50, 0.70), (0.50, 0.80),
                (0.65, 0.30), (0.65, 0.70),
            ],
        }
        radius = float(np.clip(
            3.5 if layout == "corridor" else 3.0,
            c.obs_radius_range[0],
            c.obs_radius_range[1],
        ))
        pos = [
            np.asarray(point, dtype=float) * L
            for point in templates[layout][:c.n_static_obs]
        ]
        rad = [radius for _ in pos]
        trials = 0
        total = c.n_static_obs + c.n_dynamic_obs
        while len(pos) < total and trials < 5000:
            trials += 1
            point = self.rng.uniform(0.2 * L, 0.8 * L, 2)
            candidate_radius = (
                radius if len(pos) < c.n_static_obs
                else float(self.rng.uniform(*c.obs_radius_range))
            )
            if all(
                np.linalg.norm(point - other)
                > candidate_radius + other_radius + 2.0
                for other, other_radius in zip(pos, rad)
            ):
                pos.append(point)
                rad.append(candidate_radius)
        if len(pos) != total:
            raise RuntimeError(
                f"could not place {total} obstacles for layout={layout!r}")
        positions = np.asarray(pos, dtype=float).reshape(-1, 2)
        radii = np.asarray(rad, dtype=float)
        velocities = np.zeros_like(positions)
        if c.n_dynamic_obs > 0:
            angles = self.rng.uniform(
                -math.pi, math.pi, c.n_dynamic_obs)
            velocities[-c.n_dynamic_obs:] = (
                c.dyn_obs_speed
                * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            )
        return positions, radii, velocities

    def _sample_free(self, center, spread):
        for _ in range(100):
            p = center + self.rng.uniform(-spread, spread, 2)
            p = np.clip(p, 1.0, self.cfg.arena_size - 1.0)
            if all(np.linalg.norm(p - q) > r + 2.0 for q, r in zip(self.obs_pos, self.obs_r)):
                return p
        return center

    # ---------------- 可见性 ----------------
    def _agent_sees_target(self, i: int) -> bool:
        """Legacy boolean path. Defaults remain bitwise-compatible with frozen runs."""
        c = self.cfg
        d = self.t_pos - self.p_pos[i]
        dist = np.linalg.norm(d)
        if dist > c.sense_range:
            return False
        bearing = math.atan2(d[1], d[0])
        if abs(wrap_angle(bearing - self.p_psi[i])) > c.fov_half:
            return False
        if c.visual_occlusion:
            for q, r in zip(self.obs_pos, self.obs_r):
                if segment_circle_intersect(self.p_pos[i], self.t_pos, q, r):
                    return False
        return True

    def _agent_visibility_reasons(self, i: int) -> tuple[str, ...]:
        """Return all independent reasons why agent ``i`` cannot see the target.

        Unlike the frozen boolean path this diagnostic deliberately does not
        short-circuit, allowing an event to be classified as mixed.
        """
        c = self.cfg
        d = self.t_pos - self.p_pos[i]
        dist = np.linalg.norm(d)
        reasons = []
        if dist > c.sense_range:
            reasons.append("range")
        bearing = math.atan2(d[1], d[0])
        if abs(wrap_angle(bearing - self.p_psi[i])) > c.fov_half:
            reasons.append("fov")
        if c.visual_occlusion and any(
                segment_circle_intersect(self.p_pos[i], self.t_pos, q, r)
                for q, r in zip(self.obs_pos, self.obs_r)):
            reasons.append("occlusion")
        return tuple(reasons)

    @staticmethod
    def _classify_team_visibility(
            reasons_each: list[tuple[str, ...]]) -> str:
        """Classify a team observation step without conflating failure causes."""
        if any(len(reasons) == 0 for reasons in reasons_each):
            return "visible"
        reason_sets = [set(reasons) for reasons in reasons_each]
        for reason, label in (
                ("range", "pure_range"),
                ("fov", "pure_fov"),
                ("occlusion", "pure_occlusion")):
            if all(reasons == {reason} for reasons in reason_sets):
                return label
        return "mixed"

    def _measurement_blackout_active(self) -> bool:
        start = self.cfg.measurement_blackout_start_step
        duration = self.cfg.measurement_blackout_duration_steps
        return bool(
            start is not None
            and duration > 0
            and start <= self.t < start + duration
        )

    def _update_visibility(self):
        diagnostics = self.cfg.visibility_diagnostics
        blackout_active = self._measurement_blackout_active()
        if diagnostics:
            self.visibility_reasons_each = [
                self._agent_visibility_reasons(i) for i in range(self.n)
            ]
            geometric_visible = np.array(
                [len(reasons) == 0 for reasons in self.visibility_reasons_each],
                dtype=bool,
            )
            if blackout_active:
                self.visibility_reasons_each = [
                    (*reasons, "measurement_blackout")
                    for reasons in self.visibility_reasons_each
                ]
                self.visible_each = np.zeros(self.n, dtype=bool)
                self.team_visibility_status = "measurement_blackout"
            else:
                self.visible_each = geometric_visible
                self.team_visibility_status = self._classify_team_visibility(
                    self.visibility_reasons_each)
            self.stats["agent_visibility_checks"] += self.n
            self.stats["agent_visible_checks"] += int(self.visible_each.sum())
            for reasons in self.visibility_reasons_each:
                for reason in reasons:
                    self.stats[f"agent_loss_checks_{reason}"] += 1
        else:
            # Exact frozen path: no diagnostic geometry or additional RNG use.
            self.visible_each = np.array(
                [self._agent_sees_target(i) for i in range(self.n)])
            self.visibility_reasons_each = [tuple() for _ in range(self.n)]
            self.team_visibility_status = (
                "visible" if bool(self.visible_each.any()) else "unclassified")
            if blackout_active:
                self.visible_each = np.zeros(self.n, dtype=bool)
                self.visibility_reasons_each = [
                    ("measurement_blackout",) for _ in range(self.n)
                ]
                self.team_visibility_status = "measurement_blackout"
        team_visible = bool(self.visible_each.any())
        if self.cfg.target_position_noise_std_m > 0:
            self.target_measurement_pos = np.clip(
                self.t_pos + self.obs_rng.normal(
                    0.0, self.cfg.target_position_noise_std_m, 2),
                0.0,
                self.cfg.arena_size,
            )
        else:
            # Exact legacy path: no random draw and bitwise-identical values.
            self.target_measurement_pos = self.t_pos.copy()
        if team_visible:
            if not self.prev_visible_team:
                self.stats["recaptures"] += 1
                self._just_recaptured = True
            else:
                self._just_recaptured = False
            self.t_lost = 0
            self.last_seen_pos = self.target_measurement_pos.copy()
        else:
            if self.prev_visible_team:
                self.stats["lost_events"] += 1
                if diagnostics:
                    self.stats[
                        f"loss_events_{self.team_visibility_status}"] += 1
            self._just_recaptured = False
            self.t_lost += 1
            self.stats["lost_steps"] += 1
            if diagnostics:
                self.stats[
                    f"loss_steps_{self.team_visibility_status}"] += 1
            self.stats["max_lost"] = max(self.stats["max_lost"], self.t_lost)
        for i in range(self.n):
            if self.visible_each[i]:
                rel = self.t_pos - self.p_pos[i]
                self.last_seen_rel[i] = np.concatenate([rel, self.t_vel])
        self.prev_visible_team = team_visible

    # ---------------- 观测 ----------------
    def set_belief(self, mu: np.ndarray, sigma_feat: np.ndarray, conf: float):
        """外部(GRU预测器)信念注入接口: mu为绝对位置估计, sigma_feat为3维协方差特征."""
        self.belief = dict(mu=np.asarray(mu, float), sigma=np.asarray(sigma_feat, float),
                           conf=float(conf))

    def _target_block(self, i: int) -> np.ndarray:
        """目标信息块(6维): 按 obs_mode 生成. [rx, ry, rvx, rvy, vis_flag, t_lost_norm].

        ``vis_flag`` keeps its legacy local-visibility meaning unless explicit
        team sharing is enabled.  Under sharing it means that a current target
        measurement is available locally or through the team broadcast.
        """
        c = self.cfg
        vis = bool(self.visible_each[i])
        measurement_available = bool(
            vis or (c.team_target_sharing and self.visible_each.any()))
        tl = min(self.t_lost, c.lost_cap_steps) / c.lost_cap_steps
        if measurement_available or c.obs_mode == "oracle":
            rel = self.target_measurement_pos - self.p_pos[i]
            return np.array([*(rel / c.arena_size), *(self.t_vel / c.v_max),
                             1.0 if measurement_available else 0.0, tl])
        if c.obs_mode == "zero":
            return np.array([0, 0, 0, 0, -1.0, tl])
        if c.obs_mode == "last":
            rel = self.last_seen_pos - self.p_pos[i]
            return np.array([*(rel / c.arena_size), 0, 0, -1.0, tl])
        if c.obs_mode == "belief" and self.belief is not None:
            rel = self.belief["mu"] - self.p_pos[i]
            return np.array([*(rel / c.arena_size),
                             self.belief["conf"], np.tanh(self.belief["sigma"].mean()),
                             -1.0, tl])
        return np.array([0, 0, 0, 0, -1.0, tl])

    def _get_obs(self) -> np.ndarray:
        c = self.cfg
        L = c.arena_size
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        for i in range(self.n):
            parts = [np.array([self.p_pos[i, 0] / L, self.p_pos[i, 1] / L,
                               math.cos(self.p_psi[i]), math.sin(self.p_psi[i]),
                               self.p_v[i] / c.v_max])]
            parts.append(self._target_block(i))
            for j in range(self.n):
                if j == i:
                    continue
                rel = (self.p_pos[j] - self.p_pos[i]) / L
                dv = np.array([self.p_v[j] * math.cos(self.p_psi[j]) - self.p_v[i] * math.cos(self.p_psi[i]),
                               self.p_v[j] * math.sin(self.p_psi[j]) - self.p_v[i] * math.sin(self.p_psi[i])]) / c.v_max
                parts.append(np.concatenate([rel, dv]))
            parts.append(self._nearest_obs_feat(i))
            obs[i] = np.concatenate(parts)
        return obs

    def _nearest_obs_feat(self, i: int) -> np.ndarray:
        c = self.cfg
        L = c.arena_size
        feat = np.zeros(5 * c.k_nearest_obs)
        if len(self.obs_pos) == 0:
            return feat
        d = np.linalg.norm(self.obs_pos - self.p_pos[i], axis=1) - self.obs_r
        idx = np.argsort(d)[: c.k_nearest_obs]
        for k, j in enumerate(idx):
            rel = (self.obs_pos[j] - self.p_pos[i]) / L
            feat[5 * k: 5 * k + 5] = [rel[0], rel[1], self.obs_r[j] / L,
                                      self.obs_vel[j, 0] / c.v_max, self.obs_vel[j, 1] / c.v_max]
        return feat

    def _get_state(self) -> np.ndarray:
        c = self.cfg
        L = c.arena_size
        parts = []
        for i in range(self.n):
            parts.append(np.array([self.p_pos[i, 0] / L, self.p_pos[i, 1] / L,
                                   math.cos(self.p_psi[i]), math.sin(self.p_psi[i]),
                                   self.p_v[i] / c.v_max]))
        parts.append(np.concatenate([self.t_pos / L, self.t_vel / c.v_max]))
        parts.append(np.array([min(self.t_lost, c.lost_cap_steps) / c.lost_cap_steps]))
        if len(self.obs_pos):
            d = np.linalg.norm(self.obs_pos - self.t_pos, axis=1)
            idx = np.argsort(d)[: c.k_nearest_obs]
            f = np.zeros(3 * c.k_nearest_obs)
            for k, j in enumerate(idx):
                f[3 * k: 3 * k + 3] = [self.obs_pos[j, 0] / L, self.obs_pos[j, 1] / L, self.obs_r[j] / L]
            parts.append(f)
        else:
            parts.append(np.zeros(3 * c.k_nearest_obs))
        return np.concatenate(parts).astype(np.float32)

    # ---------------- 动力学 ----------------
    def step(self, actions: np.ndarray):
        """actions: (n,2) in [-1,1] -> [a, omega]."""
        c = self.cfg
        actions = np.clip(np.asarray(actions, float).reshape(self.n, 2), -1, 1)
        self.t += 1
        # 追捕者 unicycle 积分 (式5-1/5-2)
        a = actions[:, 0] * c.a_max
        w = actions[:, 1] * c.w_max
        self.p_v = np.clip(self.p_v + a * c.dt, 0.0, c.v_max)
        self.p_psi = wrap_angle(self.p_psi + w * c.dt)
        self.p_pos = self.p_pos + np.stack([self.p_v * np.cos(self.p_psi),
                                            self.p_v * np.sin(self.p_psi)], -1) * c.dt
        self.p_pos = np.clip(self.p_pos, 0.0, c.arena_size)
        # 目标逃逸：APF / ModeSwitch / ReactiveModeSwitch
        self._step_evader()
        # 动态障碍
        self._step_dyn_obstacles()
        # 可见性与失观
        recaptured_before = self.stats["recaptures"]
        self._update_visibility()
        # 碰撞与安全统计
        col_agent, col_obs = self._collisions()
        # 奖励 (式5-19)
        reward, comps = self._reward(col_agent or col_obs)
        # 终止判定
        success = self._capture_success()
        timeout = self.t >= c.max_steps
        long_lost = self.t_lost > c.lost_cap_steps
        collided = (col_agent or col_obs) and c.collision_terminates
        done = success or timeout or collided or long_lost
        if success:
            reward += c.r_capture
            comps["terminal"] = c.r_capture
        elif long_lost:
            reward += c.r_long_lost
            comps["terminal"] = c.r_long_lost
        elif timeout:
            reward += c.r_timeout
            comps["terminal"] = c.r_timeout
        else:
            comps["terminal"] = 0.0
        collision_details = self._last_collision_details
        info = dict(
            success=success,
            timeout=timeout,
            collision=collided,
            collision_agent=bool(collided and collision_details["agent"]),
            collision_static=bool(collided and collision_details["static"]),
            collision_dynamic=bool(collided and collision_details["dynamic"]),
            contact_agent=bool(collision_details["agent"]),
            contact_static=bool(collision_details["static"]),
            contact_dynamic=bool(collision_details["dynamic"]),
            long_lost=long_lost,
            t_lost=self.t_lost,
            visible=bool(self.visible_each.any()),
            measurement_blackout_active=self._measurement_blackout_active(),
            reward_target_reference=self._target_ref_source(),
            reward_components=comps,
            stats=dict(self.stats),
        )
        if self.cfg.visibility_diagnostics:
            info["visibility_diagnostics"] = dict(
                per_agent_reasons=[
                    list(reasons) for reasons in self.visibility_reasons_each
                ],
                team_status=self.team_visibility_status,
                visual_occlusion=bool(self.cfg.visual_occlusion),
            )
        return self._get_obs(), self._get_state(), np.full(self.n, reward, np.float32), done, info

    def _step_evader(self):
        c = self.cfg
        if c.evader_type in {"mode_switch", "reactive_mode_switch"}:
            # 论文A S-A4冻结ModeSwitch及J7独立ReactiveModeSwitch扩展。
            from envs.evaders import ModeSwitchEvader
            if getattr(self, "_mode_evader", None) is None or self.t <= 1:
                reactive = c.evader_type == "reactive_mode_switch"
                self._mode_evader = ModeSwitchEvader(
                    dt=c.dt, v_max=c.speed_ratio * c.v_max, arena_size=c.arena_size,
                    dwell_range=c.evader_mode_dwell, turn_rate=c.evader_turn_rate,
                    seed=int(self.rng.integers(0, 2 ** 31 - 1)),
                    pursuer_repulsion_gain=(
                        c.evader_pursuer_repulsion_gain if reactive else 0.0),
                    pursuer_reaction_range=c.evader_pursuer_reaction_range)
            self.t_pos, self.t_vel = self._mode_evader.step(
                self.t_pos, self.obs_pos, self.obs_r,
                self.p_pos if c.evader_type == "reactive_mode_switch" else None)
            self.stats["evader_mode_switches"] = self._mode_evader.n_switches
            self.stats["evader_mode_seq"] = self._mode_evader.mode_seq_str()
            return
        L = c.arena_size
        F = np.zeros(2)
        for i in range(self.n):
            d = self.t_pos - self.p_pos[i]
            dist = max(np.linalg.norm(d), 1e-6)
            if dist < c.sense_range * 1.2:
                F += d / dist * (1.0 / max(dist - 1.0, 0.5))
        for q, r in zip(self.obs_pos, self.obs_r):
            d = self.t_pos - q
            dist = max(np.linalg.norm(d) - r, 1e-6)
            if dist < 8.0:
                F += d / max(np.linalg.norm(d), 1e-6) * (1.0 / dist) * 0.8
        margin = 8.0
        for k in range(2):
            if self.t_pos[k] < margin:
                F[k] += (margin - self.t_pos[k]) / margin
            if self.t_pos[k] > L - margin:
                F[k] -= (self.t_pos[k] - (L - margin)) / margin
        F += self.rng.normal(0, c.evader_noise, 2)
        nF = np.linalg.norm(F)
        vmax_t = c.speed_ratio * c.v_max
        self.t_vel = (F / nF * vmax_t) if nF > 1e-6 else np.zeros(2)
        self.t_pos = np.clip(self.t_pos + self.t_vel * c.dt, 0.5, L - 0.5)

    def _step_dyn_obstacles(self):
        c = self.cfg
        if c.n_dynamic_obs == 0 or len(self.obs_pos) == 0:
            return
        L = c.arena_size
        self.obs_pos = self.obs_pos + self.obs_vel * c.dt
        for j in range(len(self.obs_pos)):
            for k in range(2):
                if self.obs_pos[j, k] < self.obs_r[j] or self.obs_pos[j, k] > L - self.obs_r[j]:
                    self.obs_vel[j, k] *= -1
                    self.obs_pos[j, k] = np.clip(self.obs_pos[j, k], self.obs_r[j], L - self.obs_r[j])

    # ---------------- 奖励与判定 ----------------
    def _reward_ring_radius(self) -> float:
        configured = self.cfg.reward_target_radius
        return self.cfg.capture_radius if configured is None else configured

    def _potential(self) -> float:
        d = np.linalg.norm(self.p_pos - self._target_ref(), axis=1)
        return -abs(float(d.mean()) - self._reward_ring_radius())

    def _target_ref_source(self) -> str:
        if self.cfg.reward_target_reference == "truth":
            return "truth"
        if self.visible_each.any() or self.cfg.obs_mode == "oracle":
            return "truth"
        if self.cfg.obs_mode == "belief" and self.belief is not None:
            return "belief"
        return "last_seen"

    def _target_ref(self) -> np.ndarray:
        """奖励中的目标参考点: 可见用真值, 失观用最后观测(或信念均值)."""
        source = self._target_ref_source()
        if source == "truth":
            return self.t_pos
        if source == "belief":
            return self.belief["mu"]
        return self.last_seen_pos

    def _reward(self, collided: bool):
        c = self.cfg
        phi_prev = self._phi_prev
        phi = self._potential()
        r_dist = c.k_dist * (c.gamma_pbrs * phi - phi_prev)   # 式(5-20)
        self._phi_prev = phi
        reference = self._target_ref_source()
        reference_switch = int(reference != self._reward_ref_prev)
        self._reward_ref_prev = reference
        r_ang = 0.0
        d = np.linalg.norm(self.p_pos - self.t_pos, axis=1)
        if self.visible_each.any() and (d < 2.0 * c.capture_radius).sum() >= 2:
            gaps_std = self._gap_std()
            r_ang = -c.k_ang * gaps_std                              # 式(5-21)
        r_rec = c.k_rec if self._just_recaptured else 0.0            # 式(5-22)
        r_col = -c.k_col if collided else 0.0
        r_time = -c.k_time
        r_vis = c.k_visible if bool(self.visible_each.any()) else 0.0
        lost_fraction = min(
            float(self.t_lost) / max(c.lost_cap_steps, 1), 1.0)
        r_lost = -c.k_lost * lost_fraction
        total = r_dist + r_ang + r_rec + r_col + r_time + r_vis + r_lost
        return total, dict(
            dist=r_dist, ang=r_ang, rec=r_rec, col=r_col, time=r_time,
            visible=r_vis, lost=r_lost, potential=phi,
            potential_prev=phi_prev, reference_switch=reference_switch,
            target_reference=reference)

    def _gap_std(self) -> float:
        ang = np.arctan2(self.p_pos[:, 1] - self.t_pos[1], self.p_pos[:, 0] - self.t_pos[0])
        gaps = np.diff(np.sort(ang))
        gaps = np.append(gaps, 2 * math.pi - gaps.sum())
        return float(np.std(gaps))

    def _capture_success(self) -> bool:
        c = self.cfg
        d = np.linalg.norm(self.p_pos - self.t_pos, axis=1)
        inside = d <= c.capture_radius
        if inside.sum() < min(c.min_capture_agents, self.n):
            return False
        max_gap = sorted_gap_max(self.p_pos[inside], self.t_pos)     # 式(5-7)
        return max_gap <= c.max_gap

    def _collisions(self):
        c = self.cfg
        col_agent = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dij = np.linalg.norm(self.p_pos[i] - self.p_pos[j])
                self.stats["min_agent_dist"] = min(self.stats["min_agent_dist"], dij)
                if dij < 2 * c.pursuer_radius:
                    col_agent = True
        col_static = False
        col_dynamic = False
        n_dynamic = min(c.n_dynamic_obs, len(self.obs_pos))
        dynamic_start = len(self.obs_pos) - n_dynamic
        for i in range(self.n):
            for k, (q, r) in enumerate(zip(self.obs_pos, self.obs_r)):
                d = np.linalg.norm(self.p_pos[i] - q) - r - c.pursuer_radius
                self.stats["min_obs_dist"] = min(self.stats["min_obs_dist"], d)
                if k < dynamic_start:
                    self.stats["min_static_obs_dist"] = min(
                        self.stats["min_static_obs_dist"], d)
                else:
                    self.stats["min_dynamic_obs_dist"] = min(
                        self.stats["min_dynamic_obs_dist"], d)
                if d < 0:
                    if k < dynamic_start:
                        col_static = True
                    else:
                        col_dynamic = True
        col_obs = col_static or col_dynamic
        self._last_collision_details = dict(
            agent=bool(col_agent),
            static=bool(col_static),
            dynamic=bool(col_dynamic),
        )
        if col_agent or col_obs:
            self.stats["collisions"] += 1
        if col_agent:
            self.stats["collision_agent_steps"] += 1
        if col_static:
            self.stats["collision_static_steps"] += 1
        if col_dynamic:
            self.stats["collision_dynamic_steps"] += 1
        return col_agent, col_obs
