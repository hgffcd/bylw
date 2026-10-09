# -*- coding: utf-8 -*-
"""逃逸者策略模块（论文A S-A4）.

ModeSwitchEvader: CV/CT/急停三模式按马尔可夫链切换的机动目标.
- CV: 保持当前航向直行(巡航速度);
- CT: 固定转弯率协调转弯(入模式时随机左右);
- STOP: 急停, 速度指数衰减到 0;
三模式给出"意图速度", 再叠加 APF 障碍/墙壁排斥修正(内核与
PursuitLostEnv._step_evader 的障碍/墙壁两段逐行一致, 抽为本模块的
apf_obstacle_wall_force; 原 _step_evader 的 APF 行为本身不受影响).
可选的 pursuer_repulsion_gain>0 会额外叠加追捕者排斥，形成独立的
ReactiveModeSwitch 场景；默认增益为0，冻结 ModeSwitch 轨迹不变。
速度上限仍为 speed_ratio * v_max(由环境侧以 v_max 参数传入).

设计动机: APF 逃逸者的加速度场对滤波器而言近似单一 CV+噪声, 各信念来源
差距难以拉开; ModeSwitch 逃逸者具有离散机动模式切换, 是 IMM(CV+CT 多模型)
"用武之地"的实验设计核心(论文A E2 交叉实验).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from utils.geometry import wrap_angle

# 模式编号与精简表示字符
MODE_CV, MODE_CT, MODE_STOP = 0, 1, 2
MODE_NAMES = ("CV", "CT", "STOP")
MODE_CHARS = ("V", "T", "S")


def apf_obstacle_wall_force(t_pos: np.ndarray,
                            obs_pos: np.ndarray,
                            obs_r: np.ndarray,
                            arena_size: float,
                            obs_range: float = 8.0,
                            obs_gain: float = 0.8,
                            wall_margin: float = 8.0) -> np.ndarray:
    """障碍 + 墙壁排斥力.

    逐行复刻 PursuitLostEnv._step_evader 中的障碍排斥段与墙壁排斥段
    (默认参数即原实现中的硬编码常数), 供 ModeSwitchEvader 复用避障内核.
    注意: 不含追捕者排斥项 —— 模式机动取代了对追捕者的反应式逃逸.
    """
    t_pos = np.asarray(t_pos, float)
    F = np.zeros(2)
    for q, r in zip(obs_pos, obs_r):
        d = t_pos - q
        dist = max(np.linalg.norm(d) - r, 1e-6)
        if dist < obs_range:
            F += d / max(np.linalg.norm(d), 1e-6) * (1.0 / dist) * obs_gain
    L = arena_size
    for k in range(2):
        if t_pos[k] < wall_margin:
            F[k] += (wall_margin - t_pos[k]) / wall_margin
        if t_pos[k] > L - wall_margin:
            F[k] -= (t_pos[k] - (L - wall_margin)) / wall_margin
    return F


def apf_pursuer_force(t_pos: np.ndarray,
                      pursuer_pos: np.ndarray,
                      reaction_range: float = 30.0) -> np.ndarray:
    """追捕者排斥力，保持历史 APF 逃逸者的距离核但解耦反应半径。

    该函数只供新增 ReactiveModeSwitch 场景调用；历史 ``apf`` 分支继续
    使用原地实现，避免重构改变冻结轨迹。返回无量纲合力，调用方负责增益、
    速度尺度与限幅。
    """
    t_pos = np.asarray(t_pos, float)
    pursuer_pos = np.asarray(pursuer_pos, float)
    F = np.zeros(2)
    for p_pos in pursuer_pos:
        d = t_pos - p_pos
        dist = max(np.linalg.norm(d), 1e-6)
        if dist < reaction_range:
            F += d / dist * (1.0 / max(dist - 1.0, 0.5))
    return F


class ModeSwitchEvader:
    """CV/CT/STOP 三模式马尔可夫切换逃逸者(持有独立 rng, 种子由环境派生).

    模式动力学(意图速度, 逐步离散化, 步长 dt):
    - CV:   psi_{k+1} = psi_k,                      s = v_max;
    - CT:   psi_{k+1} = psi_k ± omega_ct * dt,      s = v_max (入模式随机定向);
    - STOP: s_{k+1} = s_k * exp(-dt / tau_stop) -> 0, psi 不变;
    意图速度 v_int = s * [cos psi, sin psi]; 命令速度
    v_cmd = clip_norm(v_int + v_max * F_apf, v_max), F_apf 为障碍/墙壁排斥.

    切换律: 停留时长 T ~ U[dwell_range] (采样后化为步数), 计满后按
    转移矩阵 P (默认无自环、对其余两模式均匀 0.5/0.5) 采样下一模式.
    """

    def __init__(self,
                 dt: float,
                 v_max: float,
                 arena_size: float,
                 dwell_range: Sequence[float] = (2.0, 4.0),
                 turn_rate: float = 0.3,
                 stop_tau: float = 0.3,
                 transition: Optional[np.ndarray] = None,
                 seed: Optional[int] = None,
                 init_heading: Optional[float] = None,
                 pursuer_repulsion_gain: float = 0.0,
                 pursuer_reaction_range: float = 30.0):
        self.dt = float(dt)
        self.v_max = float(v_max)          # = speed_ratio * v_max(环境侧传入)
        self.arena = float(arena_size)
        self.dwell_range = (float(dwell_range[0]), float(dwell_range[1]))
        self.turn_rate = float(turn_rate)
        self.stop_decay = math.exp(-self.dt / float(stop_tau))
        self.pursuer_repulsion_gain = float(pursuer_repulsion_gain)
        self.pursuer_reaction_range = float(pursuer_reaction_range)
        if self.pursuer_repulsion_gain < 0:
            raise ValueError("pursuer_repulsion_gain must be non-negative")
        if self.pursuer_reaction_range <= 0:
            raise ValueError("pursuer_reaction_range must be positive")
        self.rng = np.random.default_rng(seed)
        if transition is None:
            transition = np.array([[0.0, 0.5, 0.5],
                                   [0.5, 0.0, 0.5],
                                   [0.5, 0.5, 0.0]])
        self.P = np.asarray(transition, float)
        # 初始状态: 航向随机(或外部指定), 巡航速度, 初始模式均匀采样
        self.heading = (float(init_heading) if init_heading is not None
                        else float(self.rng.uniform(-math.pi, math.pi)))
        self.speed = self.v_max
        self.mode = int(self.rng.integers(0, 3))
        self.ct_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._dwell_left = self._sample_dwell()
        # 统计: 段序列 [(mode, steps), ...], 切换次数
        self.n_switches = 0
        self._segments = [[self.mode, 0]]

    # ---------------- 切换律 ----------------
    def _sample_dwell(self) -> int:
        lo, hi = self.dwell_range
        return max(1, int(round(self.rng.uniform(lo, hi) / self.dt)))

    def _switch(self):
        probs = self.P[self.mode]
        self.mode = int(self.rng.choice(3, p=probs / probs.sum()))
        if self.mode == MODE_CT:
            self.ct_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        if self.mode != MODE_STOP:
            self.speed = self.v_max          # 出急停/换机动模式恢复巡航速度
        self._dwell_left = self._sample_dwell()
        self.n_switches += 1
        self._segments.append([self.mode, 0])

    # ---------------- 单步推进 ----------------
    def step(self, t_pos: np.ndarray, obs_pos: np.ndarray,
             obs_r: np.ndarray,
             pursuer_pos: Optional[np.ndarray] = None) -> tuple:
        """推进一步. 输入当前目标位置与障碍, 返回 (新位置, 命令速度)."""
        t_pos = np.asarray(t_pos, float)
        if self._dwell_left <= 0:
            self._switch()
        self._dwell_left -= 1
        self._segments[-1][1] += 1
        # 模式意图速度
        if self.mode == MODE_CT:
            self.heading = float(wrap_angle(
                self.heading + self.ct_dir * self.turn_rate * self.dt))
        if self.mode == MODE_STOP:
            self.speed *= self.stop_decay
            if self.speed < 1e-3:
                self.speed = 0.0
        else:
            self.speed = self.v_max
        v_intent = self.speed * np.array([math.cos(self.heading),
                                          math.sin(self.heading)])
        # APF 障碍/墙壁排斥修正(内核复用), 限速 v_max
        F = apf_obstacle_wall_force(t_pos, obs_pos, obs_r, self.arena)
        if self.pursuer_repulsion_gain > 0 and pursuer_pos is not None:
            F += self.pursuer_repulsion_gain * apf_pursuer_force(
                t_pos, pursuer_pos, self.pursuer_reaction_range)
        v_cmd = v_intent + self.v_max * F
        n = np.linalg.norm(v_cmd)
        if n > self.v_max:
            v_cmd = v_cmd / n * self.v_max
        new_pos = np.clip(t_pos + v_cmd * self.dt, 0.5, self.arena - 0.5)
        return new_pos, v_cmd

    # ---------------- 统计输出 ----------------
    def mode_seq_str(self) -> str:
        """精简模式序列: 如 'V32|T25|S40' (字母=模式, 数字=停留步数)."""
        return "|".join(f"{MODE_CHARS[m]}{s}" for m, s in self._segments if s > 0)

    def mode_step_counts(self) -> dict:
        """各模式累计步数 {'CV': n, 'CT': n, 'STOP': n}."""
        out = {name: 0 for name in MODE_NAMES}
        for m, s in self._segments:
            out[MODE_NAMES[m]] += s
        return out
