# -*- coding: utf-8 -*-
"""预测器在环的 MAPPO 训练入口(M2 point 基线 / M4 belief 模式的训练循环地基).

以 train_baseline.py 为骨架复制改造, 新增:
- 冻结 GRU 预测器在环: 每步用各环境的目标观测历史推理得到 (mu, sigma_feat, conf),
  经 env.set_belief() 注入后重新取观测(env.obs_mode 必须为 belief);
- train.belief_variant 五种变体:
    point  -> 同样使用预测器 mu, 但 sigma_feat 置零、conf 恒 1.0(无置信度点预测基线, M2);
    belief -> 完整 (mu, sigma_feat, conf) 注入(M4, 下一步再接 ConfidenceGAT);
    no_covariance -> 保留 mu/conf，sigma_feat 置零(J4);
    fixed_confidence -> 保留 mu/sigma_feat，conf 恒 1.0(J4);
    oracle_confidence -> 保留IMM的mu/sigma_feat，只用真实估计误差生成置信度上界(T2消融);
- 每个并行环境独立维护预测器观测历史窗口与失观 rollout 缓存, 回合 reset 时重置;
- 预测器全程 torch.no_grad() 冻结; 评估函数同样走注入流程.
- 论文A S-A3: --belief-source {gru,imm,kf,cv,hybrid}(默认 gru, 行为不变). 非 gru 源
  不加载 GRU, 用 FilterBeliefManager(belief/imm_filter|kf_single|cv_extrap) 驱动
  set_belief, 三元组口径与 GRU 版严格一致(mu 米制 / sigma_feat 归一化 / conf 同参);
  hybrid=G1门后启用的IMM先验+GRU残差，独立checkpoint由
  scripts/train_hybrid_residual.py训练.

基础设施与 train_baseline.py 同等(2026-07-27 对齐):
- recipe 合并(cfg["recipe"] -> configs/recipes/<name>.yaml, env/train 段合并 + eval_env
  + recipe_version)、课程退火(train.anneal, 只作用训练env)、CBF 安全层在环(train.use_cbf,
  rollout 与评估动作过 train_baseline.apply_safety)、eval_cfg 分离(评估恒用 env终值+
  anneal终点+eval_env覆盖, 预测器在环流程同样走 eval_cfg 环境)、eval_log.csv +
  best_eval.pt(按确定性评估成功率)、runs/registry.csv 登记、--seed/--iters/--resume
  CLI 覆盖、tag 带种子;
- 预测器 checkpoint 兼容 train_predictor.py --temp-scale 的 'temperature' 键:
  load 时 pop 出来, 推理时按其口径 Sigma -> T*Sigma 对预测方差乘 T.

三方约定(读码确认):
- env.set_belief(mu, sigma_feat, conf): mu 为绝对坐标[m](env 内做 mu-p_pos 再 /L,
  奖励 _target_ref 直接用 mu); sigma_feat 文档约定 3 维协方差特征, env 侧仅消费
  tanh(sigma_feat.mean()); conf 为标量, 直接进观测;
- GRU 预测器(belief/gru_predictor.py)输入输出均为归一化单位(pos/arena, vel/8),
  输出对角方差 2 维 —— 本文件适配为 sigma_feat=[var_x, var_y, 0](对角近似, 交叉项置0),
  保持归一化尺度以匹配 rollout_lost 的 q0 膨胀与 confidence 的 l2 标度;
- 失观期自回归递推直接调用 rollout_lost(勿重写): 每次失观事件从最后可见窗口一次性
  rollout lost_cap_steps+1 步并缓存, 之后按 t_lost 索引; 伪观测行 [mu,0,0,-1]
  同步压入滚动窗口(镜像 rollout_lost 内部约定), 保证重捕获后窗口连续.

用法(本机 rl_env):
    python train_belief.py --config configs/c1_point_r4.yaml --seed 1
    python train_belief.py --config configs/c1_belief_r4.yaml --seed 1 --resume <ckpt>

输出: runs/<tag>/ 下保存 config 快照、训练曲线 csv、best/best_eval/final checkpoint
(与基线一致), 并追加 runs/registry.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import yaml

from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
from algos.mappo import MAPPO, RolloutBuffer
from belief.cv_extrap import CVExtrapolator
from belief.gru_predictor import GRUGaussianPredictor
from belief.hybrid import HybridBeliefManager, load_hybrid_checkpoint
from belief.imm_filter import IMMFilter
from belief.kf_single import KFSingleCV
from belief.predictor_io import load_predictor_checkpoint
from safety.factory import build_safety_layer
from train_baseline import apply_safety  # 复用基线CBF动作滤波(内部用_agent_vel)
from utils.reacquisition import build_reacquisition_controller

VEL_NORM = 8.0  # 与 scripts/train_predictor.py build_samples 的 vel/8.0 保持一致

# 论文A S-A3: --belief-source 非 gru 时的滤波器工厂
FILTER_SOURCES = {"imm": IMMFilter, "kf": KFSingleCV, "cv": CVExtrapolator}
BELIEF_VARIANTS = (
    "point", "belief", "no_covariance", "fixed_confidence",
    "oracle_confidence",
)


def select_belief_features(
    sigma_feat: np.ndarray,
    conf: float,
    variant: str,
) -> tuple[np.ndarray, float]:
    """Apply a fixed-dimensional belief-feature ablation.

    All variants keep the observation dimension and mean estimate unchanged:
    ``belief`` uses the complete historical channel; ``no_covariance`` removes
    the scalar covariance-summary input; ``fixed_confidence`` removes confidence
    dynamics; ``point`` removes both uncertainty inputs; and
    ``oracle_confidence`` is passed through here because its scalar is generated
    by the manager from true estimation error.
    """
    if variant not in BELIEF_VARIANTS:
        raise ValueError(f"unknown belief variant: {variant!r}")
    sigma = np.asarray(sigma_feat, dtype=float).copy()
    if sigma.shape != (3,):
        raise ValueError(f"sigma_feat must have shape (3,), got {sigma.shape}")
    confidence = float(conf)
    if variant in {"point", "no_covariance"}:
        sigma.fill(0.0)
    if variant in {"point", "fixed_confidence"}:
        confidence = 1.0
    return sigma, confidence


def oracle_confidence_from_error(
    estimated_pos_m: np.ndarray,
    true_pos_m: np.ndarray,
    scale_m: float,
) -> float:
    """Convert true instantaneous position error to an oracle confidence.

    This helper is deliberately isolated so the target state can only enter the
    explicitly named upper-bound ablation, never the shared IMM estimate.
    """
    scale = float(scale_m)
    if scale <= 0:
        raise ValueError("oracle confidence scale must be positive")
    error_m = float(np.linalg.norm(
        np.asarray(estimated_pos_m, dtype=float)
        - np.asarray(true_pos_m, dtype=float)))
    return float(np.exp(-0.5 * (error_m / scale) ** 2))


def load_predictor(ckpt_path: str) -> tuple[torch.nn.Module, float | None]:
    """加载并冻结概率预测器，同时兼容旧版裸 GRU state_dict。

    新版自描述 checkpoint 可承载 GRU 或概率 Transformer；二者暴露相同的
    ``forward/rollout_lost/confidence`` 接口，因此闭环信念管理逻辑无需分叉。
    """
    model, temperature, _ = load_predictor_checkpoint(
        ckpt_path, map_location="cpu", freeze=True)
    return model, temperature


class BeliefManager:
    """单个并行环境的预测器在环状态机.

    状态 = 滚动观测窗口(承载 GRU 隐状态, 与预测器训练时的定长窗口条件一致)
         + 失观事件的 rollout 缓存(mus/vars, 按 t_lost 索引).
    窗口特征行 = [t_x/L, t_y/L, vx/8, vy/8, vis_flag], 同 build_samples 约定;
    失观步压入伪观测行 [mu_x, mu_y, 0, 0, -1](同 rollout_lost 内部约定).
    temperature: train_predictor.py --temp-scale 拟合的校准温度, 消费口径为
    预测方差 var -> var*T(缓存的 rollout 方差同样在消费时乘 T).
    """

    def __init__(self, predictor: torch.nn.Module, variant: str,
                 arena: float, lost_cap: int, seq_len: int = 20,
                 temperature: float | None = None):
        assert variant in BELIEF_VARIANTS, variant
        self.pred = predictor
        self.variant = variant
        self.arena = float(arena)
        self.lost_cap = int(lost_cap)
        self.seq_len = int(seq_len)
        self.temp = temperature
        self.window: torch.Tensor | None = None   # (1, seq_len, 5)
        self.lost_mus: torch.Tensor | None = None  # (1, S, 2)
        self.lost_vars: torch.Tensor | None = None
        self.lost_t0 = 0

    def _row(self, pos: np.ndarray, vel: np.ndarray, flag: float) -> torch.Tensor:
        f = np.concatenate([pos / self.arena, vel / VEL_NORM, [flag]])
        return torch.as_tensor(f, dtype=torch.float32).view(1, 1, 5)

    def _push(self, row: torch.Tensor):
        self.window = torch.cat([self.window[:, 1:], row], dim=1)

    def reset(self, env: PursuitLostEnv) -> np.ndarray:
        """env.reset() 之后调用: 重置窗口与失观缓存, 注入信念并返回观测."""
        # 回合初以 last_seen_pos(reset 时即目标真实位置)铺满窗口作为热身
        row = self._row(env.last_seen_pos, env.t_vel, 1.0)
        self.window = row.repeat(1, self.seq_len, 1)
        self.lost_mus = self.lost_vars = None
        self.lost_t0 = 0
        return self.inject(env)

    @torch.no_grad()
    def inject(self, env: PursuitLostEnv) -> np.ndarray:
        """env.step()/reset() 之后调用: 更新历史 -> 推理 -> set_belief -> 重取观测."""
        visible = bool(env.visible_each.any())
        t_lost = int(env.t_lost)
        if visible:
            # 可见: 用真实观测更新窗口(=更新隐状态), 取一步预测
            self.lost_mus = self.lost_vars = None
            measured_pos = getattr(
                env, "target_measurement_pos", env.t_pos)
            self._push(self._row(
                measured_pos, env.t_vel, 1.0))
            mu_k, var_k, _ = self.pred(self.window)
            mu, var = mu_k[0, 0], var_k[0, 0]
        else:
            if self.lost_mus is None:
                # 新失观事件: 从最后可见窗口一次性自回归递推并缓存(式5-11)
                self.lost_t0 = max(t_lost, 1)
                self.lost_mus, self.lost_vars = self.pred.rollout_lost(
                    self.window, steps=self.lost_cap + 1)
            idx = min(max(t_lost - self.lost_t0, 0), self.lost_mus.shape[1] - 1)
            mu, var = self.lost_mus[0, idx], self.lost_vars[0, idx]
            self._push(torch.cat([mu, torch.zeros(2), torch.tensor([-1.0])]).view(1, 1, 5))
        if self.temp is not None:
            var = var * self.temp        # 温度校准: Sigma -> T*Sigma
        mu_abs = np.clip(mu.numpy() * self.arena, 0.0, self.arena)
        v = var.numpy()
        sigma_feat = np.array([v[0], v[1], 0.0])  # [Sxx, Syy, Sxy], 对角近似
        conf = GRUGaussianPredictor.confidence(t_lost, float(v.sum()))
        sigma_feat, conf = select_belief_features(
            sigma_feat, conf, self.variant)
        env.set_belief(mu_abs, sigma_feat, conf)
        return env._get_obs()


class FilterBeliefManager:
    """滤波驱动的信念状态机(论文A S-A3, --belief-source imm/kf/cv), 与 BeliefManager 同构.

    滤波器(belief/imm_filter|kf_single|cv_extrap)全程绝对米制: 状态 [x,y,vx,vy],
    可见步 update(目标真实观测位置 env.t_pos), 失观步 predict(dt) 纯传播递推.
    注入 set_belief 的三元组口径与 GRU 版严格一致:
    - mu: 绝对米制坐标, clip 到 [0, arena](同 BeliefManager.inject);
    - sigma_feat = [var_x, var_y, 0]: **归一化单位** = 米制位置方差 / arena^2
      (GRU 输出方差本就在 pos/arena 尺度, 此处除 arena^2 对齐量纲);
    - conf = GRUGaussianPredictor.confidence(t_lost, tr(Sigma_norm)):
      λ1=0.02/λ2=0.05 与 GRU 版同源同参; t_lost 传步数——与 GRU 版调用口径逐字一致
      (即 exp(-λ1·t_lost_steps - λ2·trΣ_norm), λ1 已隐含吸收 dt 标度).
    """

    def __init__(self, source: str, variant: str, arena: float,
                 filter_kwargs: dict | None = None,
                 oracle_confidence_scale_m: float = 5.0):
        assert variant in BELIEF_VARIANTS, variant
        assert source in FILTER_SOURCES, source
        self.filter_kwargs = dict(filter_kwargs or {})
        self.filt = FILTER_SOURCES[source](**self.filter_kwargs)
        self.variant = variant
        self.arena = float(arena)
        self.oracle_confidence_scale_m = float(oracle_confidence_scale_m)
        if self.oracle_confidence_scale_m <= 0:
            raise ValueError("oracle_confidence_scale_m must be positive")

    def reset(self, env: PursuitLostEnv) -> np.ndarray:
        """env.reset() 之后调用: 用 last_seen_pos(reset 时即目标真实位置)初始化滤波器."""
        self.filt.reset(env.last_seen_pos.copy(), env.t_vel.copy())
        return self.inject(env)

    def inject(self, env: PursuitLostEnv) -> np.ndarray:
        """env.step()/reset() 之后调用: 滤波递推 -> set_belief -> 重取观测."""
        dt = env.cfg.dt
        if bool(env.visible_each.any()):
            measured_pos = getattr(
                env, "target_measurement_pos", env.t_pos)
            mu, Sigma, _, _ = self.filt.update(
                measured_pos.copy(), dt)
        else:
            mu, Sigma, _ = self.filt.predict(dt)
        mu_abs = np.clip(mu[:2], 0.0, self.arena)
        var_norm = np.diag(Sigma)[:2] / (self.arena ** 2)
        sigma_feat = np.array([var_norm[0], var_norm[1], 0.0])  # 对角近似, 交叉项置0
        conf = GRUGaussianPredictor.confidence(
            int(env.t_lost), float(var_norm.sum()))
        if self.variant == "oracle_confidence":
            # T2 upper-bound ablation only: reveal the true instantaneous
            # estimation error to the confidence channel, while keeping the
            # shared IMM mean/covariance unchanged.  The target state is never
            # inserted into mu or sigma.
            conf = oracle_confidence_from_error(
                mu_abs, env.t_pos, self.oracle_confidence_scale_m)
        sigma_feat, conf = select_belief_features(
            sigma_feat, conf, self.variant)
        env.set_belief(mu_abs, sigma_feat, conf)
        return env._get_obs()


def make_envs(env_cfg: EnvConfig, n_env: int, base_seed: int):
    envs = []
    for k in range(n_env):
        cfg_k = EnvConfig(**{**asdict(env_cfg), "seed": base_seed + k})
        envs.append(PursuitLostEnv(cfg_k))
    return envs


def parse_random_measurement_blackout(train_cfg, env_cfg):
    """Validate an opt-in per-episode short measurement-blackout curriculum.

    Values are configured in seconds and converted once to inclusive integer
    step ranges.  A missing block returns ``None`` and consumes no RNG, which
    preserves every historical training path.
    """
    raw = train_cfg.get("random_measurement_blackout")
    if not raw:
        return None
    probability = float(raw.get("probability", 0.0))
    start_s = list(raw.get("start_s", []))
    duration_s = list(raw.get("duration_s", []))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "train.random_measurement_blackout.probability must be in [0, 1]")
    if len(start_s) != 2 or not 0.0 <= float(start_s[0]) <= float(start_s[1]):
        raise ValueError(
            "train.random_measurement_blackout.start_s must be [min, max] "
            "with 0 <= min <= max")
    if len(duration_s) != 2 or not 0.0 < float(duration_s[0]) <= float(duration_s[1]):
        raise ValueError(
            "train.random_measurement_blackout.duration_s must be [min, max] "
            "with 0 < min <= max")
    dt = float(env_cfg.dt)
    start_steps = tuple(int(round(float(value) / dt)) for value in start_s)
    duration_steps = tuple(
        max(1, int(round(float(value) / dt))) for value in duration_s
    )
    if duration_steps[1] > int(env_cfg.lost_cap_steps):
        raise ValueError(
            "random measurement-blackout duration must not exceed "
            "env.lost_cap_steps")
    return {
        "probability": probability,
        "start_steps": start_steps,
        "duration_steps": duration_steps,
    }


def apply_random_measurement_blackout(env, rng, spec):
    """Sample and apply one episode's blackout, returning ``(start, duration)``."""
    if spec is None:
        return (
            env.cfg.measurement_blackout_start_step,
            env.cfg.measurement_blackout_duration_steps,
        )
    if spec["probability"] <= 0.0 or rng.random() >= spec["probability"]:
        start, duration = None, 0
    else:
        start = int(rng.integers(
            spec["start_steps"][0], spec["start_steps"][1] + 1))
        duration = int(rng.integers(
            spec["duration_steps"][0], spec["duration_steps"][1] + 1))
    env.cfg.measurement_blackout_start_step = start
    env.cfg.measurement_blackout_duration_steps = duration
    return start, duration


def evaluate(
    agent,
    env_cfg,
    mgr_factory,
    episodes=20,
    seed=10_000,
    device="cpu",
    use_cbf=False,
    safety_version="legacy",
    safety_params=None,
    reacquisition_params=None,
):
    """mgr_factory: env -> BeliefManager|FilterBeliefManager(信念来源无关的评估流程)."""
    stats = dict(success=0, collision=0, timeout=0, long_lost=0, ep_len=[], recap=0, lost_ev=0)
    layer = (
        build_safety_layer(env_cfg, safety_version, safety_params)
        if use_cbf else None
    )
    for e in range(episodes):
        env = PursuitLostEnv(EnvConfig(**{**asdict(env_cfg), "seed": seed + e}))
        env.reset()
        mgr = mgr_factory(env)
        obs = mgr.reset(env)
        reacquisition = build_reacquisition_controller(reacquisition_params)
        done, t = False, 0
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(o)
                act = torch.tanh(mean)[0].cpu().numpy()
            if reacquisition is not None:
                act, _ = reacquisition.apply(env, act)
            if layer is not None:
                act = apply_safety(env, act, layer)
            _, _, _, done, info = env.step(act)
            obs = mgr.inject(env)
            t += 1
        for k in ("success", "collision", "timeout", "long_lost"):
            stats[k] += int(info[k])
        stats["ep_len"].append(t)
        stats["recap"] += info["stats"]["recaptures"]
        stats["lost_ev"] += info["stats"]["lost_events"]
    n = episodes
    return dict(success_rate=stats["success"] / n, collision_rate=stats["collision"] / n,
                timeout_rate=stats["timeout"] / n, long_lost_rate=stats["long_lost"] / n,
                mean_len=float(np.mean(stats["ep_len"])),
                recapture_ratio=stats["recap"] / max(stats["lost_ev"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/c2_point.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=None, help="覆盖配置中的训练迭代数")
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置中的训练种子(多种子纪律)")
    ap.add_argument("--resume", default=None, help="从checkpoint继续训练(课程分阶段用)")
    ap.add_argument("--predictor-ckpt", default=None, help="覆盖配置中的预测器checkpoint路径")
    ap.add_argument("--hybrid-ckpt", default=None,
                    help="覆盖配置中的IMM-GRU残差checkpoint路径")
    ap.add_argument("--belief-variant", default=None, choices=BELIEF_VARIANTS,
                    help="覆盖配置中的信念注入变体")
    ap.add_argument("--belief-source", default=None,
                    choices=["gru", "imm", "kf", "cv", "hybrid"],
                    help="信念来源(论文A S-A3): gru=预测器在环(默认); imm/kf/cv=滤波驱动"
                         "(不加载GRU); hybrid=IMM先验+G1门后GRU残差")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 配方合并: cfg["recipe"] 指向 configs/recipes/<name>.yaml, 具体config可覆盖配方默认
    if "recipe" in cfg:
        with open(os.path.join("configs", "recipes", cfg["recipe"] + ".yaml"),
                  encoding="utf-8") as f:
            rec = yaml.safe_load(f)
        cfg["env"] = {**rec.get("env", {}), **cfg.get("env", {})}
        cfg["train"] = {**rec.get("train", {}), **cfg.get("train", {})}
        # Preserve recipe-level strict evaluation fields while allowing an
        # experiment to override individual values.
        cfg["eval_env"] = {
            **rec.get("eval_env", {}),
            **(cfg.get("eval_env", {}) or {}),
        }
        cfg["recipe_version"] = rec.get("version", cfg["recipe"])
    env_cfg = EnvConfig(**cfg["env"])
    tr = cfg["train"]
    if args.iters:
        tr["iters"] = args.iters
    if args.seed is not None:
        tr["seed"] = args.seed
    if args.predictor_ckpt:
        tr["predictor_ckpt"] = args.predictor_ckpt
    if args.hybrid_ckpt:
        tr["hybrid_ckpt"] = args.hybrid_ckpt
    if args.belief_variant:
        tr["belief_variant"] = args.belief_variant
    if args.belief_source:
        tr["belief_source"] = args.belief_source
    if args.log_every:
        tr["log_every"] = args.log_every
    if args.eval_every:
        tr["eval_every"] = args.eval_every

    if env_cfg.obs_mode != "belief":
        raise SystemExit(f"train_belief.py 要求 env.obs_mode=belief, 当前为 {env_cfg.obs_mode!r}")
    variant = tr.get("belief_variant", "belief")
    oracle_confidence_scale_m = float(
        tr.get("oracle_confidence_scale_m", 5.0))
    seq_len = int(tr.get("predictor_seq_len", 20))
    source = tr.get("belief_source", "gru")
    if source == "hybrid" and variant not in {"point", "belief"}:
        raise SystemExit(
            "hybrid目前只支持point/belief；J4新消融仅预注册用于IMM")
    if source == "hybrid":
        hybrid_path = tr.get("hybrid_ckpt")
        if not hybrid_path:
            raise SystemExit("--belief-source hybrid需要train.hybrid_ckpt或--hybrid-ckpt")
        residual_model, hybrid_meta = load_hybrid_checkpoint(hybrid_path)
        filter_kwargs = dict((tr.get("filter_params") or {}).get("imm", {}))
        trained_filter = dict(hybrid_meta.get("filter_params") or {})
        if trained_filter and trained_filter != filter_kwargs:
            raise SystemExit(
                f"hybrid checkpoint IMM参数{trained_filter}与配置{filter_kwargs}不一致")
        hybrid_seq_len = int(hybrid_meta.get("seq_len", seq_len))
        residual_temperature = float(
            hybrid_meta.get("residual_temperature", 1.0))
        print(f"hybrid loaded: {hybrid_path} seq_len={hybrid_seq_len} "
              f"temperature={residual_temperature:.4f} "
              f"IMM params={filter_kwargs or 'defaults'}")

        def mgr_factory(e):
            return HybridBeliefManager(
                residual_model, variant, e.cfg.arena_size,
                seq_len=hybrid_seq_len,
                residual_temperature=residual_temperature,
                filter_kwargs=filter_kwargs)
    elif source == "gru":
        predictor, pred_temp = load_predictor(tr["predictor_ckpt"])
        print(f"predictor loaded: {tr['predictor_ckpt']} (horizon={predictor.horizon}) "
              f"variant={variant} temperature={pred_temp}")

        def mgr_factory(e):
            return BeliefManager(predictor, variant, e.cfg.arena_size,
                                 e.cfg.lost_cap_steps, seq_len, pred_temp)
    elif source in FILTER_SOURCES:
        # 参数按来源分组，保证同一个配置可由 --belief-source 切换 CV/KF/IMM。
        # 未配置时保持历史默认值，旧 run 可被原样复现。
        filter_kwargs = dict((tr.get("filter_params") or {}).get(source, {}))
        print(f"belief source = {source} (滤波驱动, 不加载GRU预测器) "
              f"variant={variant} params={filter_kwargs or 'defaults'}")

        def mgr_factory(e):
            return FilterBeliefManager(source, variant, e.cfg.arena_size,
                                       filter_kwargs=filter_kwargs,
                                       oracle_confidence_scale_m=
                                       oracle_confidence_scale_m)
    else:
        raise SystemExit(f"未知 belief_source: {source!r}")

    # 评估环境固定为最终判据: env终值 + anneal终点 + eval_env覆盖, 训练期永不变动
    eval_fields = dict(cfg["env"])
    if tr.get("anneal"):
        for f_name, (v0, v1) in tr["anneal"]["fields"].items():
            eval_fields[f_name] = float(v1)
    eval_fields.update(cfg.get("eval_env", {}) or {})
    eval_cfg = EnvConfig(**eval_fields)

    src_sfx = f"_{source}" if source != "gru" else ""  # 非gru源入tag, E1各行可区分
    tag = (cfg.get("tag", "run") + src_sfx + f"_s{tr['seed']}" + time.strftime("_%m%d_%H%M"))
    out = os.path.join("runs", tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    # 登记: (recipe版本, 判据参数, 种子, 承接run) —— 与 train_baseline 同文件同格式
    reg = os.path.join("runs", "registry.csv")
    new_reg = not os.path.exists(reg)
    with open(reg, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_reg:
            w.writerow(["time", "run_dir", "recipe", "seed", "resumed_from",
                        "eval_criteria", "obs_mode", "iters"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), out,
                    cfg.get("recipe_version", "-"), tr["seed"], args.resume or "-",
                    f"rho={eval_cfg.capture_radius};gap={eval_cfg.max_gap:.4f};"
                    f"k_col={eval_cfg.k_col};col_term={eval_cfg.collision_terminates}",
                    eval_cfg.obs_mode, tr["iters"]])

    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    envs = make_envs(env_cfg, tr["n_env"], tr["seed"] * 1000)
    n_env, n_agent = tr["n_env"], env_cfg.n_pursuers
    probe = envs[0]
    agent = MAPPO(probe.obs_dim, probe.state_dim, probe.act_dim, n_agent,
                  hidden=tr["hidden"], lr=tr["lr"], clip=tr["clip"],
                  epochs=tr["epochs"], minibatch=tr["minibatch"],
                  ent_coef=tr["ent_coef"], device=args.device,
                  actor_lr=tr.get("actor_lr"), critic_lr=tr.get("critic_lr"),
                  actor_encoder=tr.get("actor_encoder", "mlp"),
                  actor_encoder_kwargs=tr.get("actor_encoder_kwargs"),
                  policy_anchor_coef=float(tr.get("policy_anchor_coef", 0.0)))
    if args.resume:
        agent.load(args.resume)
        print(f"resumed from {args.resume}")
    policy_anchor_coef = float(tr.get("policy_anchor_coef", 0.0))
    if policy_anchor_coef > 0:
        if not args.resume:
            raise ValueError("policy_anchor_coef requires --resume checkpoint")
        agent.set_policy_anchor()
        print(
            f"policy anchor frozen from resumed actor: coef={policy_anchor_coef:g}")

    episode_mix = dict(tr.get("episode_mix") or {})
    easy_probability = float(episode_mix.get("easy_probability", 0.0))
    easy_fields = dict(episode_mix.get("easy_fields") or {})
    if not 0.0 <= easy_probability <= 1.0:
        raise ValueError("train.episode_mix.easy_probability must be in [0, 1]")
    hard_fields = {name: getattr(env_cfg, name) for name in easy_fields}
    mix_rngs = [
        np.random.default_rng(int(tr["seed"]) * 100_000 + k)
        for k in range(n_env)
    ]
    random_blackout = parse_random_measurement_blackout(tr, env_cfg)
    blackout_rngs = [
        np.random.default_rng(int(tr["seed"]) * 100_000 + 50_000 + k)
        for k in range(n_env)
    ]
    if random_blackout is not None:
        print(
            "random measurement blackout training: "
            f"probability={random_blackout['probability']:.3f} "
            f"start_steps={random_blackout['start_steps']} "
            f"duration_steps={random_blackout['duration_steps']}",
            flush=True,
        )

    def reset_training_env(index):
        env = envs[index]
        use_easy = bool(easy_fields) and mix_rngs[index].random() < easy_probability
        for name, hard_value in hard_fields.items():
            setattr(env.cfg, name, easy_fields[name] if use_easy else hard_value)
        apply_random_measurement_blackout(
            env, blackout_rngs[index], random_blackout)
        return env.reset()

    mgrs = [mgr_factory(e) for e in envs]
    obs_list = []
    for k, (e, m) in enumerate(zip(envs, mgrs)):
        reset_training_env(k)
        obs_list.append(m.reset(e))
    obs = np.stack(obs_list)
    state = np.stack([e._get_state() for e in envs])

    best = -1.0
    log_path = os.path.join(out, "train_log.csv")
    train_log_columns = [
        "iter", "ep_ret", "success", "pi_loss", "v_loss", "entropy",
    ]
    if policy_anchor_coef > 0:
        train_log_columns.append("anchor_kl")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(train_log_columns)

    ep_ret = np.zeros(n_env)
    ret_hist, succ_hist = [], []
    best_eval = -1.0
    retain_earliest_eval_tie = bool(tr.get("retain_earliest_eval_tie", False))
    # CBF安全层(M5): use_cbf=true 时训练rollout与评估全程经安全投影
    use_cbf = bool(tr.get("use_cbf", False))
    safety_version = str(tr.get("safety_version", "legacy"))
    safety_params = dict(tr.get("safety_params") or {})
    cbf_layers = [
        build_safety_layer(env_cfg, safety_version, safety_params)
        for _ in range(n_env)
    ] if use_cbf else None
    reacquisition_params = dict(tr.get("reacquisition_controller") or {})
    reacquisition_controllers = [
        build_reacquisition_controller(reacquisition_params)
        for _ in range(n_env)
    ]
    cbf_correction_penalty = float(tr.get("cbf_correction_penalty", 0.0))
    if cbf_correction_penalty < 0:
        raise ValueError("train.cbf_correction_penalty must be non-negative")
    if tr.get("eval_resume_at_start", False):
        if not args.resume:
            raise ValueError("eval_resume_at_start requires --resume checkpoint")
        initial_eval = evaluate(
            agent,
            eval_cfg,
            mgr_factory,
            episodes=tr["eval_episodes"],
            device=args.device,
            use_cbf=use_cbf,
            safety_version=safety_version,
            safety_params=safety_params,
            reacquisition_params=reacquisition_params,
        )
        print(
            f"    eval@resume(rho={eval_cfg.capture_radius:.1f},"
            f"gap={eval_cfg.max_gap:.2f},k_col={eval_cfg.k_col:.0f},"
            f"term={eval_cfg.collision_terminates}): {initial_eval}",
            flush=True,
        )
        ev_path = os.path.join(out, "eval_log.csv")
        with open(ev_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iter", "success", "collision", "timeout", "long_lost",
                "mean_len",
            ])
            writer.writerow([
                0, initial_eval["success_rate"],
                initial_eval["collision_rate"], initial_eval["timeout_rate"],
                initial_eval["long_lost_rate"], initial_eval["mean_len"],
            ])
        best_eval = initial_eval["success_rate"]
        agent.save(os.path.join(out, "best_eval.pt"))
        print(f"    best_eval.pt <- resume {best_eval:.2f}", flush=True)
    # 课程退火: train.anneal = {start, end, fields: {字段: [初值, 终值]}}
    # 每迭代对 EnvConfig 浮点字段线性插值, 只作用训练env(eval_cfg 固定为终值判据)
    anneal = tr.get("anneal")
    for it in range(1, tr["iters"] + 1):
        if anneal:
            frac = min(max((it - anneal["start"]) / max(anneal["end"] - anneal["start"], 1), 0.0), 1.0)
            for f_name, (v0, v1) in anneal["fields"].items():
                val = float(v0) + (float(v1) - float(v0)) * frac
                for e in envs:
                    setattr(e.cfg, f_name, val)
        buf = RolloutBuffer(tr["rollout"], n_env, n_agent,
                            probe.obs_dim, probe.state_dim, probe.act_dim,
                            agent.device)
        for _ in range(tr["rollout"]):
            o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            s = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            act, logp, val = agent.act(o, s)
            act_np = act.cpu().numpy()
            rew = np.zeros(n_env, np.float32)
            dn = np.zeros(n_env, np.float32)
            for k, env in enumerate(envs):
                a_exec = act_np[k]
                if reacquisition_controllers[k] is not None:
                    a_exec, _ = reacquisition_controllers[k].apply(env, a_exec)
                safety_diag = None
                if use_cbf:
                    if cbf_correction_penalty > 0:
                        a_exec, safety_diag = apply_safety(
                            env, a_exec, cbf_layers[k], return_diagnostics=True)
                    else:
                        a_exec = apply_safety(env, a_exec, cbf_layers[k])
                ob, st, r, done, info = env.step(a_exec)
                shaped_reward = float(r[0])
                if safety_diag is not None:
                    shaped_reward -= cbf_correction_penalty * float(
                        safety_diag.get("correction_norm", 0.0))
                rew[k] = shaped_reward
                ep_ret[k] += shaped_reward
                if done:
                    succ_hist.append(float(info["success"]))
                    ret_hist.append(ep_ret[k])
                    ep_ret[k] = 0.0
                    reset_training_env(k)
                    ob = mgrs[k].reset(env)       # 重置预测器状态并注入
                    st = env._get_state()
                    if reacquisition_controllers[k] is not None:
                        reacquisition_controllers[k].reset()
                    dn[k] = 1.0
                else:
                    ob = mgrs[k].inject(env)      # 更新历史->注入->重取观测
                obs[k], state[k] = ob, st
            buf.add(o, s, act, logp,
                    torch.as_tensor(rew, device=agent.device),
                    torch.as_tensor(dn, device=agent.device), val)
        with torch.no_grad():
            last_val = agent.critic(torch.as_tensor(state, dtype=torch.float32,
                                                    device=agent.device)).squeeze(-1)
        info_upd = agent.update(buf, last_val, gamma=tr["gamma"], lam=tr["lam"])
        if it % tr["log_every"] == 0:
            mret = float(np.mean(ret_hist[-50:])) if ret_hist else 0.0
            msucc = float(np.mean(succ_hist[-50:])) if succ_hist else 0.0
            print(f"[{it:4d}] ret={mret:8.2f} succ={msucc:.2f} "
                  f"pi={info_upd['pi_loss']:.3f} v={info_upd['v_loss']:.3f}"
                  + (f" anchor_kl={info_upd['anchor_kl']:.5f}"
                     if policy_anchor_coef > 0 else ""))
            with open(log_path, "a", newline="") as f:
                row = [
                    it, mret, msucc, info_upd["pi_loss"],
                    info_upd["v_loss"], info_upd["entropy"],
                ]
                if policy_anchor_coef > 0:
                    row.append(info_upd["anchor_kl"])
                csv.writer(f).writerow(row)
            if msucc >= best and it > tr["iters"] // 10:
                best = msucc
                agent.save(os.path.join(out, "best.pt"))
        if it % tr["eval_every"] == 0:
            ev = evaluate(
                agent,
                eval_cfg,
                mgr_factory,
                episodes=tr["eval_episodes"],
                device=args.device,
                use_cbf=use_cbf,
                safety_version=safety_version,
                safety_params=safety_params,
                reacquisition_params=reacquisition_params,
            )
            print(f"    eval(rho={eval_cfg.capture_radius:.1f},gap={eval_cfg.max_gap:.2f},"
                  f"k_col={eval_cfg.k_col:.0f},term={eval_cfg.collision_terminates}): {ev}",
                  flush=True)
            ev_path = os.path.join(out, "eval_log.csv")
            new_ev = not os.path.exists(ev_path)
            with open(ev_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_ev:
                    w.writerow(["iter", "success", "collision", "timeout", "long_lost", "mean_len"])
                w.writerow([it, ev["success_rate"], ev["collision_rate"],
                            ev["timeout_rate"], ev["long_lost_rate"], ev["mean_len"]])
            # 按验收口径(确定性+最终判据)保存最优checkpoint——rolling best与验收不对齐
            improves = ev["success_rate"] > best_eval
            ties_and_latest_wins = (
                ev["success_rate"] == best_eval
                and not retain_earliest_eval_tie
            )
            if improves or ties_and_latest_wins:
                best_eval = ev["success_rate"]
                agent.save(os.path.join(out, "best_eval.pt"))
                print(f"    best_eval.pt <- {best_eval:.2f}", flush=True)
    agent.save(os.path.join(out, "final.pt"))
    print("done ->", out)


if __name__ == "__main__":
    main()
