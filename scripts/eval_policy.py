# -*- coding: utf-8 -*-
"""验收评估(唯一口径): 确定性策略 x 固定评估配置 x >=100回合 x >=3评估种子.

    python scripts/eval_policy.py --ckpt runs/<run>/final.pt \
        --config configs/eval_final_c0.yaml --episodes 100 --eval-seeds 3
输出各种子成功率与 mean±std, 以及失败回合分类(碰撞/超时/长期失观).

obs_mode 与信念源属方法侧变量, 验收须与训练一致: belief 训练行须配
--obs-mode belief --belief-source {gru,imm,kf,cv,hybrid}
(gru 另需 --predictor-ckpt，hybrid另需--hybrid-ckpt),
信念管理器在环逐步 set_belief 注入(复用 train_belief, 时序同 train_belief.evaluate);
与 --cbf/--lost-bins/--bins-csv 正交组合可用; 不带新参数时行为逐位不变.
"""
import argparse
import csv
import math
import os
import sys
from dataclasses import fields as dataclass_fields
from numbers import Real

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
from algos.networks import build_gaussian_actor
from safety.factory import SAFETY_VERSIONS, build_safety_layer
from train_baseline import apply_safety
from utils.reacquisition import build_reacquisition_controller
from utils.geometry import sorted_gap_max


VISIBILITY_STAT_FIELDS = [
    "agent_visibility_checks",
    "agent_visible_checks",
    "agent_loss_checks_range",
    "agent_loss_checks_fov",
    "agent_loss_checks_occlusion",
    "agent_loss_checks_measurement_blackout",
    "loss_steps_pure_range",
    "loss_steps_pure_fov",
    "loss_steps_pure_occlusion",
    "loss_steps_mixed",
    "loss_steps_measurement_blackout",
    "loss_events_pure_range",
    "loss_events_pure_fov",
    "loss_events_pure_occlusion",
    "loss_events_mixed",
    "loss_events_measurement_blackout",
]

REWARD_COMPONENT_FIELDS = [
    "dist", "ang", "rec", "col", "time", "visible", "lost", "terminal",
]

CAPTURE_SENSITIVITY_FIELDS = [
    f"capture_r{radius}_k{agents}_step"
    for radius in (4, 5, 6)
    for agents in (1, 2, 3)
]

BELIEF_MECHANISM_FIELDS = [
    "belief_lost_steps",
    "suspected_region_coverage95_rate",
    "duplicate_coverage_rate",
    "mean_belief_confidence",
    "mean_target_attention",
    "attention_confidence_slope",
    "attention_confidence_correlation",
    "reacquisition_events",
    "mean_reacquisition_time_s",
    "max_reacquisition_time_s",
]


def _pairwise_duplicate_coverage(env, center, angle_threshold=math.pi / 6):
    """Fraction of pursuer pairs occupying nearly the same bearing sector."""
    bearings = np.arctan2(
        env.p_pos[:, 1] - center[1],
        env.p_pos[:, 0] - center[0],
    )
    duplicates = []
    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            difference = abs(math.atan2(
                math.sin(bearings[i] - bearings[j]),
                math.cos(bearings[i] - bearings[j]),
            ))
            duplicates.append(float(difference < angle_threshold))
    return float(np.mean(duplicates)) if duplicates else 0.0


def _attention_statistics(confidences, target_attentions):
    if len(confidences) < 2 or not np.isfinite(target_attentions).any():
        return float("nan"), float("nan")
    x = np.asarray(confidences, float)
    y = np.asarray(target_attentions, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.var(x) <= 1e-12 or np.var(y) <= 1e-12:
        return float("nan"), float("nan")
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    correlation = float(np.corrcoef(x, y)[0, 1])
    return slope, correlation


def effective_training_eval_fields(run_cfg):
    """Reconstruct the environment used by train_belief's periodic evaluation."""
    env_fields = dict(run_cfg.get("env") or {})
    anneal_fields = (
        ((run_cfg.get("train") or {}).get("anneal") or {}).get("fields") or {}
    )
    for name, values in anneal_fields.items():
        env_fields[name] = float(values[-1])
    env_fields.update(run_cfg.get("eval_env") or {})
    return env_fields


def _same_env_value(left, right):
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        try:
            return tuple(left) == tuple(right)
        except TypeError:
            return False
    if isinstance(left, Real) and isinstance(right, Real):
        return bool(np.isclose(left, right, rtol=1e-9, atol=1e-9))
    return left == right


def managed_env_mismatches(run_cfg, requested_env_fields):
    """Return protocol differences other than method-side obs_mode and episode seed."""
    trained = EnvConfig(**effective_training_eval_fields(run_cfg))
    requested = EnvConfig(**requested_env_fields)
    mismatches = []
    for item in dataclass_fields(EnvConfig):
        name = item.name
        if name in {"seed", "obs_mode"}:
            continue
        left, right = getattr(trained, name), getattr(requested, name)
        if not _same_env_value(left, right):
            mismatches.append((name, left, right))
    return mismatches


def _capture_sensitivity_hits(env):
    """Return frozen phase-2 first-passage predicates for the current state.

    One/two-agent conditions are interception/cooperative-capture auxiliaries
    and deliberately do not claim angular enclosure.  Three-agent conditions
    retain the main ``max_gap <= pi`` encirclement requirement.
    """
    distances = np.linalg.norm(env.p_pos - env.t_pos, axis=1)
    hits = {}
    for radius in (4, 5, 6):
        inside = distances <= float(radius)
        count = int(inside.sum())
        hits[f"capture_r{radius}_k1_step"] = count >= 1
        hits[f"capture_r{radius}_k2_step"] = count >= 2
        hits[f"capture_r{radius}_k3_step"] = bool(
            count >= 3
            and sorted_gap_max(env.p_pos[inside], env.t_pos) <= math.pi
        )
    return hits


def _instantaneous_clearances(env):
    """Return physical center distances and surface clearances at this step.

    The environment's episode statistics retain running minima.  P0 collision
    replay additionally needs the step at which the margin becomes negative,
    so this helper deliberately uses the same radii and dynamic-obstacle split
    as ``PursuitLostEnv._collisions`` without introducing a new threshold.
    """
    agent_distances = [
        float(np.linalg.norm(env.p_pos[i] - env.p_pos[j]))
        for i in range(env.n)
        for j in range(i + 1, env.n)
    ]
    min_agent_center = min(agent_distances, default=float("nan"))
    min_agent_clearance = (
        min_agent_center - 2.0 * env.cfg.pursuer_radius
        if np.isfinite(min_agent_center) else float("nan")
    )
    n_dynamic = min(env.cfg.n_dynamic_obs, len(env.obs_pos))
    dynamic_start = len(env.obs_pos) - n_dynamic
    static_clearances = []
    dynamic_clearances = []
    for pursuer in env.p_pos:
        for index, (center, radius) in enumerate(
                zip(env.obs_pos, env.obs_r)):
            clearance = float(
                np.linalg.norm(pursuer - center)
                - float(radius) - env.cfg.pursuer_radius
            )
            target = (
                dynamic_clearances if index >= dynamic_start
                else static_clearances
            )
            target.append(clearance)
    all_obstacle_clearances = static_clearances + dynamic_clearances
    return {
        "min_agent_center_distance": min_agent_center,
        "min_agent_clearance": min_agent_clearance,
        "min_obstacle_clearance": min(
            all_obstacle_clearances, default=float("nan")),
        "min_static_obstacle_clearance": min(
            static_clearances, default=float("nan")),
        "min_dynamic_obstacle_clearance": min(
            dynamic_clearances, default=float("nan")),
    }


def run_block(
    actor,
    env_fields,
    episodes,
    seed_base,
    use_cbf=False,
    collect=False,
    mgr_factory=None,
    safety_version="legacy",
    safety_params=None,
    reacquisition_params=None,
    zero_policy_actions=False,
    collect_steps=False,
):
    """mgr_factory: env -> BeliefManager|FilterBeliefManager(复用 train_belief), 非 None 时
    信念源在环: reset/step 后更新管理器 -> set_belief -> 重取观测(同 train_belief.evaluate
    时序); None 时行为不变(belief 模式下信念恒 None, 观测信念块退化)."""
    st = dict(success=0, collision=0, timeout=0, long_lost=0, lens=[])
    records = []                         # collect=True 时: 逐回合(max_lost步数,结局,长度)
    safety_records = []
    step_records = []
    for e in range(episodes):
        env = PursuitLostEnv(EnvConfig(**{**env_fields, "seed": seed_base + e}))
        layer = (
            build_safety_layer(env.cfg, safety_version, safety_params)
            if use_cbf else None
        )
        obs, _ = env.reset()
        mgr = None
        if mgr_factory is not None:
            mgr = mgr_factory(env)
            obs = mgr.reset(env)         # 重置管理器状态并注入(env.reset 之后调用)
        reacquisition = build_reacquisition_controller(reacquisition_params)
        done, t, info = False, 0, {}
        safety_steps = 0
        safety_corrected_steps = 0
        safety_correction_norm_sum = 0.0
        safety_correction_norm_max = 0.0
        belief_coverages = []
        duplicate_coverages = []
        belief_confidences = []
        target_attentions = []
        reacquisition_times = []
        current_lost_steps = 0
        reacquisition_active_steps = 0
        reacquisition_scouts = set()
        reward_component_sums = {
            name: 0.0 for name in REWARD_COMPONENT_FIELDS
        }
        reward_sum = 0.0
        reward_reference_switches = 0
        reward_reference_steps = {
            "truth": 0, "belief": 0, "last_seen": 0,
        }
        first_passage = {
            name: -1 for name in CAPTURE_SENSITIVITY_FIELDS
        }
        blackout_steps = 0
        previous_safety_summary = (
            layer.summary()
            if collect_steps and layer is not None
            and hasattr(layer, "summary") else {}
        )
        while not done:
            with torch.no_grad():
                mu, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
                act = torch.tanh(mu).numpy()
            if zero_policy_actions:
                act = np.zeros_like(act)
            nominal_act = np.asarray(act, float).copy()
            if env.t_lost > 0 and env.belief is not None:
                belief_mean = np.asarray(env.belief["mu"], float)
                belief_sigma = np.asarray(env.belief["sigma"], float)
                marginal_variance = np.maximum(
                    belief_sigma[:2], 1e-12) * env.cfg.arena_size ** 2
                error = np.abs(env.t_pos - belief_mean)
                belief_coverages.append(float(
                    (error <= 1.96 * np.sqrt(marginal_variance)).all()))
                duplicate_coverages.append(
                    _pairwise_duplicate_coverage(env, belief_mean))
                belief_confidences.append(float(env.belief["conf"]))
                encoder = getattr(actor, "encoder", None)
                attention = (
                    getattr(getattr(encoder, "gat", None),
                            "last_attention", None)
                    if encoder is not None else None
                )
                if attention is None:
                    target_attentions.append(float("nan"))
                else:
                    target_attentions.append(float(
                        attention[..., 1].float().mean().cpu()))
            if reacquisition is not None:
                act, reacquisition_diag = reacquisition.apply(env, act)
                if reacquisition_diag.get("active", False):
                    reacquisition_active_steps += 1
                    reacquisition_scouts.add(int(
                        reacquisition_diag.get("scout_index", -1)))
            safety_diag = {}
            safety_counter_deltas = {}
            if layer is not None:
                act, safety_diag = apply_safety(
                    env, act, layer, return_diagnostics=True)
                safety_steps += 1
                safety_corrected_steps += int(
                    safety_diag.get("corrected", False))
                correction_norm = float(
                    safety_diag.get("correction_norm", 0.0))
                safety_correction_norm_sum += correction_norm
                safety_correction_norm_max = max(
                    safety_correction_norm_max, correction_norm)
                if collect_steps and hasattr(layer, "summary"):
                    current_safety_summary = layer.summary()
                    for name in (
                        "calls", "qp_attempts", "qp_failures", "fallbacks",
                        "verified_safe", "output_unsafe",
                    ):
                        safety_counter_deltas[name] = int(
                            current_safety_summary.get(name, 0)
                            - previous_safety_summary.get(name, 0)
                        )
                    previous_safety_summary = current_safety_summary
            executed_act = np.asarray(act, float).copy()
            obs, _, reward, done, info = env.step(executed_act)
            components = info["reward_components"]
            reward_sum += float(reward[0])
            for name in REWARD_COMPONENT_FIELDS:
                reward_component_sums[name] += float(
                    components.get(name, 0.0))
            reward_reference_switches += int(
                components.get("reference_switch", 0))
            reference = str(info.get(
                "reward_target_reference", "unknown"))
            reward_reference_steps[reference] = (
                reward_reference_steps.get(reference, 0) + 1)
            blackout_steps += int(
                info.get("measurement_blackout_active", False))
            hits = _capture_sensitivity_hits(env)
            for name, hit in hits.items():
                if hit and first_passage[name] < 0:
                    first_passage[name] = int(env.t)
            if collect_steps:
                clearances = _instantaneous_clearances(env)
                row = {
                    "episode": e,
                    "env_seed": seed_base + e,
                    "step": int(env.t),
                    "visible": int(info["visible"]),
                    "team_visibility_status": env.team_visibility_status,
                    "measurement_blackout_active": int(
                        info.get("measurement_blackout_active", False)),
                    "t_lost": int(info["t_lost"]),
                    "reward": float(reward[0]),
                    "reward_target_reference": reference,
                    "reference_switch": int(
                        components.get("reference_switch", 0)),
                    "potential": float(components.get("potential", 0.0)),
                    "potential_prev": float(
                        components.get("potential_prev", 0.0)),
                    "target_x": float(env.t_pos[0]),
                    "target_y": float(env.t_pos[1]),
                    "success": int(info["success"]),
                    "timeout": int(info["timeout"]),
                    "collision": int(info["collision"]),
                    "collision_agent": int(info["collision_agent"]),
                    "collision_static": int(info["collision_static"]),
                    "collision_dynamic": int(info["collision_dynamic"]),
                    "contact_agent": int(info.get("contact_agent", False)),
                    "contact_static": int(info.get("contact_static", False)),
                    "contact_dynamic": int(info.get("contact_dynamic", False)),
                    "long_lost": int(info["long_lost"]),
                    "safety_called": int(layer is not None),
                    "safety_corrected": int(
                        safety_diag.get("corrected", False)),
                    "safety_correction_norm": float(
                        safety_diag.get("correction_norm", 0.0)),
                    "safety_nominal_min_clearance": float(
                        safety_diag.get("nominal_min_clearance", float("nan"))),
                    "safety_output_min_clearance": float(
                        safety_diag.get("output_min_clearance", float("nan"))),
                    "safety_output_horizon_physical_clearance": float(
                        safety_diag.get(
                            "output_horizon_physical_clearance", float("nan"))),
                    "safety_output_next_step_physical_clearance": float(
                        safety_diag.get(
                            "output_next_step_physical_clearance", float("nan"))),
                    "safety_solver_status": str(
                        safety_diag.get("solver_status", "not_used")),
                    "safety_solver_residual": float(
                        safety_diag.get("solver_residual", float("nan"))),
                    "safety_fallback": int(
                        safety_diag.get("fallback", False)),
                    "safety_emergency_tier": str(
                        safety_diag.get("emergency_tier", "not_used")),
                    **{
                        f"safety_{name}_delta": int(
                            safety_counter_deltas.get(name, 0))
                        for name in (
                            "calls", "qp_attempts", "qp_failures",
                            "fallbacks", "verified_safe", "output_unsafe",
                        )
                    },
                    **clearances,
                    **{
                        f"reward_{name}": float(
                            components.get(name, 0.0))
                        for name in REWARD_COMPONENT_FIELDS
                    },
                    **{name.replace("_step", ""): int(hit)
                       for name, hit in hits.items()},
                }
                for index in range(env.n):
                    row.update({
                        f"p{index}_x": float(env.p_pos[index, 0]),
                        f"p{index}_y": float(env.p_pos[index, 1]),
                        f"p{index}_v": float(env.p_v[index]),
                        f"p{index}_psi": float(env.p_psi[index]),
                        f"p{index}_nominal_a": float(
                            nominal_act[index, 0]),
                        f"p{index}_nominal_w": float(
                            nominal_act[index, 1]),
                        f"p{index}_executed_a": float(
                            executed_act[index, 0]),
                        f"p{index}_executed_w": float(
                            executed_act[index, 1]),
                    })
                dynamic_start = len(env.obs_pos) - min(
                    env.cfg.n_dynamic_obs, len(env.obs_pos))
                for index, (center, radius, velocity) in enumerate(zip(
                        env.obs_pos, env.obs_r, env.obs_vel)):
                    row.update({
                        f"obs{index}_x": float(center[0]),
                        f"obs{index}_y": float(center[1]),
                        f"obs{index}_radius": float(radius),
                        f"obs{index}_vx": float(velocity[0]),
                        f"obs{index}_vy": float(velocity[1]),
                        f"obs{index}_dynamic": int(index >= dynamic_start),
                    })
                step_records.append(row)
            if info["visible"]:
                if current_lost_steps > 0:
                    reacquisition_times.append(
                        current_lost_steps * env.cfg.dt)
                current_lost_steps = 0
            else:
                current_lost_steps += 1
            if mgr is not None:
                obs = mgr.inject(env)    # 更新历史/滤波 -> set_belief -> 重取观测
            t += 1
        for k in ("success", "collision", "timeout", "long_lost"):
            st[k] += int(info[k])
        st["lens"].append(t)
        if collect:
            stats = info["stats"]
            record = dict(
                episode=e,
                env_seed=seed_base + e,
                max_lost=int(stats["max_lost"]),
                success=int(info["success"]),
                collision=int(info["collision"]),
                collision_agent=int(info["collision_agent"]),
                collision_static=int(info["collision_static"]),
                collision_dynamic=int(info["collision_dynamic"]),
                long_lost=int(info["long_lost"]),
                ep_len=t,
                evader_mode_switches=int(
                    stats.get("evader_mode_switches", 0)),
                evader_mode_seq=str(
                    stats.get("evader_mode_seq", "")),
                reacquisition_active_steps=int(reacquisition_active_steps),
                reacquisition_scout_count=len(
                    {index for index in reacquisition_scouts if index >= 0}),
                reward_sum=reward_sum,
                reward_reference_switches=reward_reference_switches,
                reward_reference_truth_steps=reward_reference_steps["truth"],
                reward_reference_belief_steps=reward_reference_steps["belief"],
                reward_reference_last_seen_steps=reward_reference_steps[
                    "last_seen"],
                measurement_blackout_steps=blackout_steps,
            )
            record.update({
                f"reward_{name}_sum": value
                for name, value in reward_component_sums.items()
            })
            record.update(first_passage)
            record.update({
                key: int(stats.get(key, 0))
                for key in VISIBILITY_STAT_FIELDS
            })
            attention_slope, attention_correlation = _attention_statistics(
                belief_confidences, target_attentions)
            record.update({
                "belief_lost_steps": len(belief_coverages),
                "suspected_region_coverage95_rate": (
                    float(np.mean(belief_coverages))
                    if belief_coverages else float("nan")),
                "duplicate_coverage_rate": (
                    float(np.mean(duplicate_coverages))
                    if duplicate_coverages else float("nan")),
                "mean_belief_confidence": (
                    float(np.mean(belief_confidences))
                    if belief_confidences else float("nan")),
                "mean_target_attention": (
                    float(np.nanmean(target_attentions))
                    if np.isfinite(target_attentions).any()
                    else float("nan")),
                "attention_confidence_slope": attention_slope,
                "attention_confidence_correlation": attention_correlation,
                "reacquisition_events": len(reacquisition_times),
                "mean_reacquisition_time_s": (
                    float(np.mean(reacquisition_times))
                    if reacquisition_times else float("nan")),
                "max_reacquisition_time_s": (
                    float(np.max(reacquisition_times))
                    if reacquisition_times else float("nan")),
            })
            records.append(record)
        if layer is not None:
            if hasattr(layer, "summary"):
                safety_summary = layer.summary()
            else:
                safety_summary = {
                    "safety_version": "legacy",
                    "calls": int(layer.call_count),
                    "corrections": int(layer.correction_count),
                    "correction_rate": float(layer.correction_rate),
                }
            safety_summary.update({
                "wrapper_steps": safety_steps,
                "wrapper_corrected_steps": safety_corrected_steps,
                "wrapper_correction_rate": (
                    safety_corrected_steps / max(safety_steps, 1)
                ),
                "wrapper_mean_correction_norm": (
                    safety_correction_norm_sum / max(safety_steps, 1)
                ),
                "wrapper_max_correction_norm": (
                    safety_correction_norm_max
                ),
            })
            safety_records.append({
                "episode": e,
                "env_seed": seed_base + e,
                "success": int(info["success"]),
                "collision": int(info["collision"]),
                "collision_agent": int(info["collision_agent"]),
                "collision_static": int(info["collision_static"]),
                "collision_dynamic": int(info["collision_dynamic"]),
                "long_lost": int(info["long_lost"]),
                "timeout": int(info["timeout"]),
                "ep_len": t,
                "min_agent_center_distance": float(
                    info["stats"].get("min_agent_dist", float("nan"))),
                "min_agent_clearance": float(
                    info["stats"].get("min_agent_dist", float("nan"))
                    - 2.0 * env.cfg.pursuer_radius),
                "min_obstacle_clearance": float(
                    info["stats"].get("min_obs_dist", float("nan"))),
                "min_static_obstacle_clearance": float(
                    info["stats"].get(
                        "min_static_obs_dist", float("nan"))),
                "min_dynamic_obstacle_clearance": float(
                    info["stats"].get(
                        "min_dynamic_obs_dist", float("nan"))),
                "reacquisition_active_steps": int(
                    reacquisition_active_steps),
                "reacquisition_scout_count": len(
                    {index for index in reacquisition_scouts if index >= 0}),
                "reacquisition_events": len(reacquisition_times),
                "mean_reacquisition_time_s": (
                    float(np.mean(reacquisition_times))
                    if reacquisition_times else float("nan")),
                "max_reacquisition_time_s": (
                    float(np.max(reacquisition_times))
                    if reacquisition_times else float("nan")),
                **safety_summary,
            })
    n = episodes
    out = dict(success=st["success"] / n, collision=st["collision"] / n,
               timeout=st["timeout"] / n, long_lost=st["long_lost"] / n,
               mean_len=float(np.mean(st["lens"])))
    if collect:
        out["episodes"] = records
    if use_cbf:
        out["safety_episodes"] = safety_records
    if collect_steps:
        out["steps"] = step_records
    return out


def write_safety_stats_csv(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    preferred = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "collision_agent", "collision_static", "collision_dynamic",
        "long_lost", "timeout", "ep_len",
        "min_agent_center_distance", "min_agent_clearance",
        "min_obstacle_clearance", "min_static_obstacle_clearance",
        "min_dynamic_obstacle_clearance",
        "safety_version", "calls", "corrections",
        "correction_rate", "nominal_unsafe", "nominal_unsafe_rate",
        "qp_attempts", "qp_failures", "fallbacks", "fallback_rate",
        "verified_safe", "output_unsafe", "mean_correction_norm",
        "max_correction_norm", "minimum_solver_residual",
        "minimum_predicted_clearance",
        "output_horizon_physical_unsafe",
        "output_next_step_physical_unsafe",
        "minimum_horizon_physical_clearance",
        "minimum_next_step_physical_clearance", "wrapper_steps",
        "wrapper_corrected_steps", "wrapper_correction_rate",
        "wrapper_mean_correction_norm", "wrapper_max_correction_norm",
    ]
    extra = sorted({
        key for record in records for key in record if key not in preferred
    })
    fieldnames = preferred + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_visibility_stats_csv(path, records):
    """Write episode-level causal visibility counts for thesis T2/T4."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "long_lost", "ep_len", "max_lost", *VISIBILITY_STAT_FIELDS,
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: record[name] for name in columns}
            for record in records
        )


def write_belief_mechanism_csv(path, records):
    """Write episode-level T2 belief coverage and attention diagnostics."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "long_lost", "ep_len", "max_lost", *BELIEF_MECHANISM_FIELDS,
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: record[name] for name in columns}
            for record in records
        )


def write_reward_stats_csv(path, records):
    """Write episode-level reward decomposition and reference-switch audit."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "long_lost", "ep_len", "reward_sum",
        *[f"reward_{name}_sum" for name in REWARD_COMPONENT_FIELDS],
        "reward_reference_switches", "reward_reference_truth_steps",
        "reward_reference_belief_steps", "reward_reference_last_seen_steps",
        "measurement_blackout_steps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: record[name] for name in columns}
            for record in records
        )


def write_capture_sensitivity_csv(path, records):
    """Write per-episode first-passage steps for the pre-registered criteria."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "long_lost", "ep_len", *CAPTURE_SENSITIVITY_FIELDS,
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: record[name] for name in columns}
            for record in records
        )


def write_step_telemetry_csv(path, records):
    """Write detailed action/visibility/reward traces for causal diagnosis."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    preferred = [
        "eval_seed", "episode", "env_seed", "step", "visible",
        "team_visibility_status", "measurement_blackout_active", "t_lost",
        "success", "timeout", "collision", "collision_agent",
        "collision_static", "collision_dynamic", "contact_agent",
        "contact_static", "contact_dynamic", "long_lost",
        "safety_called", "safety_corrected", "safety_correction_norm",
        "safety_nominal_min_clearance", "safety_output_min_clearance",
        "safety_output_horizon_physical_clearance",
        "safety_output_next_step_physical_clearance",
        "safety_solver_status", "safety_solver_residual",
        "safety_fallback", "safety_emergency_tier",
        "safety_calls_delta", "safety_qp_attempts_delta",
        "safety_qp_failures_delta", "safety_fallbacks_delta",
        "safety_verified_safe_delta", "safety_output_unsafe_delta",
        "min_agent_center_distance", "min_agent_clearance",
        "min_obstacle_clearance", "min_static_obstacle_clearance",
        "min_dynamic_obstacle_clearance",
        "reward", "reward_target_reference", "reference_switch",
        "potential", "potential_prev",
        *[f"reward_{name}" for name in REWARD_COMPONENT_FIELDS],
        "target_x", "target_y",
    ]
    extra = sorted({
        key for record in records for key in record if key not in preferred
    })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + extra)
        writer.writeheader()
        writer.writerows(records)


def parse_lost_bins(spec, dt):
    """秒切点 -> 左闭右开分箱边界. 返回 (edges_s, edges_steps), 均含 0 与 +inf 端点."""
    cuts = [float(x) for x in spec.split(",") if x.strip()]
    if not cuts or any(c <= 0 for c in cuts) or \
            any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"--lost-bins 需为严格递增的正数秒切点: {spec!r}")
    edges_s = [0.0] + cuts + [float("inf")]
    edges_steps = [0] + [int(round(c / dt)) for c in cuts] + [float("inf")]
    if any(b <= a for a, b in zip(edges_steps, edges_steps[1:])):
        raise ValueError(f"--lost-bins 切点经 dt={dt} 换算步数后出现重合: {spec!r}")
    return edges_s, edges_steps


def accumulate_bins(records, edges_steps):
    """逐回合记录按 max_lost(步)落入 [edges_steps[i], edges_steps[i+1]) 累计."""
    k = len(edges_steps) - 1
    acc = dict(n=np.zeros(k, int), success=np.zeros(k, int),
               collision=np.zeros(k, int), long_lost=np.zeros(k, int),
               capture_len_sum=np.zeros(k, float))
    for r in records:
        i = k - 1
        for j in range(k):
            if edges_steps[j] <= r["max_lost"] < edges_steps[j + 1]:
                i = j
                break
        acc["n"][i] += 1
        acc["success"][i] += r["success"]
        acc["collision"][i] += r["collision"]
        acc["long_lost"][i] += r["long_lost"]
        if r["success"]:
            acc["capture_len_sum"][i] += r["ep_len"]
    return acc


def merge_bins(accs):
    total = {k: sum(a[k] for a in accs) for k in accs[0]}
    return total


def bin_table_rows(total, edges_s):
    """-> [(bin_low_s, bin_high_s, n, success率, mean_capture_len, collision率, long_lost率)]"""
    rows = []
    for i in range(len(edges_s) - 1):
        n = int(total["n"][i])
        succ = int(total["success"][i])
        rows.append((edges_s[i], edges_s[i + 1], n,
                     succ / n if n else float("nan"),
                     total["capture_len_sum"][i] / succ if succ else float("nan"),
                     int(total["collision"][i]) / n if n else float("nan"),
                     int(total["long_lost"][i]) / n if n else float("nan")))
    return rows


def _fmt(v, spec):
    return "-" if (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def print_bin_table(rows, n_seeds, episodes):
    print(f"\n[lost-bins 总表] {n_seeds}评估种子 x {episodes}回合 合并 "
          f"(按回合最大连续团队失观时长分箱, 左闭右开)")
    print(f"{'bin[s)':>12} {'n':>5} {'success':>8} {'cap_len':>8} "
          f"{'collision':>9} {'long_lost':>9}")
    for lo, hi, n, sr, cl, cr, lr in rows:
        hi_s = "inf" if np.isinf(hi) else f"{hi:g}"
        label = f"[{lo:g},{hi_s})"
        print(f"{label:>12} {n:>5} {_fmt(sr, '{:.3f}'):>8} "
              f"{_fmt(cl, '{:.1f}'):>8} {_fmt(cr, '{:.3f}'):>9} "
              f"{_fmt(lr, '{:.3f}'):>9}")


def write_bins_csv(path, rows):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bin_low_s", "bin_high_s", "n", "success",
                    "mean_capture_len", "collision", "long_lost"])

        def cell(v, nd):
            return "" if (isinstance(v, float) and np.isnan(v)) else round(v, nd)
        for lo, hi, n, sr, cl, cr, lr in rows:
            w.writerow([f"{lo:g}", "inf" if np.isinf(hi) else f"{hi:g}", n,
                        cell(sr, 4), cell(cl, 2), cell(cr, 4), cell(lr, 4)])


def write_mode_stats_csv(path, records):
    """Write per-episode ModeSwitch traces for downstream occupancy/dwell analysis."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    columns = [
        "eval_seed", "episode", "env_seed", "success", "collision",
        "long_lost", "ep_len", "max_lost", "evader_mode_switches",
        "evader_mode_seq",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows({name: row[name] for name in columns} for row in records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="固定评估配置(configs/eval_final_*.yaml)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--eval-seed-base", type=int, default=100_000,
                    help="首个评估种子基址；正式协议默认100000，开发诊断必须改用独立基址")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--cbf", action="store_true", help="启用CBF安全层(本文方法含M5层)")
    ap.add_argument("--safety-version", default="legacy",
                    choices=SAFETY_VERSIONS,
                    help="安全层版本；legacy保持冻结复现，dynamic_v2仅用于J2b及后续显式协议")
    ap.add_argument("--safety-horizon-s", type=float, default=None,
                    help="dynamic_v2预测时域，缺省使用版本默认值")
    ap.add_argument("--safety-buffer", type=float, default=None,
                    help="dynamic_v2固定几何安全缓冲[m]")
    ap.add_argument("--safety-activation-distance", type=float, default=None,
                    help="dynamic_v2约束激活距离[m]")
    ap.add_argument(
        "--safety-emergency-ranking",
        choices=["horizon", "safe_prefix"],
        default=None,
        help="dynamic_v2 emergency候选排序；horizon复现保留的revision2.2",
    )
    ap.add_argument("--safety-stats-csv", default=None,
                    help="逐回合安全修正、求解残差、回退和碰撞来源统计")
    ap.add_argument(
        "--visibility-stats-csv",
        default=None,
        help="T2/T4逐回合range/fov/occlusion失观原因统计；"
             "评估配置必须启用visibility_diagnostics",
    )
    ap.add_argument(
        "--belief-mechanism-csv",
        default=None,
        help="T2逐回合信念95%%区域覆盖、重复覆盖、重捕获时间和"
             "注意力—置信度响应；要求belief-source在环",
    )
    ap.add_argument(
        "--reward-stats-csv",
        default=None,
        help="逐回合奖励分量、势函数参考源切换和blackout步数统计",
    )
    ap.add_argument(
        "--capture-sensitivity-csv",
        default=None,
        help="逐回合4/5/6m与1/2/3-of-3首次满足步统计",
    )
    ap.add_argument(
        "--step-telemetry-csv",
        default=None,
        help="逐步可见模式、奖励、势函数、名义/安全层动作和位置遥测",
    )
    ap.add_argument("--obs-mode", default=None, choices=["oracle", "zero", "last", "belief"],
                    help="覆盖评估配置的obs_mode(观测处理属方法侧, 验收须与训练一致)")
    ap.add_argument(
        "--obstacle-layout",
        choices=["random", "corridor", "cluster", "narrow_passage"],
        default=None,
        help="J3显式OOD障碍布局覆盖",
    )
    ap.add_argument(
        "--target-position-noise-std-m",
        type=float,
        default=None,
        help="J3可见目标位置测量噪声标准差[m]",
    )
    ap.add_argument("--sense-range", type=float, default=None,
                    help="J3感知距离OOD覆盖[m]")
    ap.add_argument(
        "--visual-occlusion",
        choices=["true", "false"],
        default=None,
        help="P4可见性因果评估：显式启用或禁用障碍视线遮挡；不改变碰撞几何",
    )
    ap.add_argument(
        "--initial-visibility-reference-sense-range",
        type=float,
        default=None,
        help="P4 matched因果评估：仅用于生成共同初始已探测几何的参考距离[m]",
    )
    ap.add_argument(
        "--initial-visibility-reference-visual-occlusion",
        choices=["true", "false"],
        default=None,
        help="P4 matched因果评估：仅用于共同初始几何采样的参考遮挡开关",
    )
    ap.add_argument(
        "--team-target-sharing",
        action="store_true",
        help=("任一追捕者观测到目标后，向全队广播当前目标"
              "位置、速度和测量可用标志；局部FOV/遮挡遥测不变"),
    )
    ap.add_argument("--fov-half-deg", type=float, default=None,
                    help="J3视场半角OOD覆盖[deg]")
    ap.add_argument("--capture-radius", type=float, default=None,
                    help="显式捕获半径敏感性覆盖[m]")
    ap.add_argument("--min-capture-agents", type=int, default=None,
                    help="显式k-of-N捕获人数覆盖")
    ap.add_argument("--max-gap-deg", type=float, default=None,
                    help="显式围捕最大角缺口覆盖[deg]")
    ap.add_argument("--measurement-blackout-start-s", type=float, default=None,
                    help="受控全队目标测量中断开始时间[s]")
    ap.add_argument("--measurement-blackout-duration-s", type=float, default=None,
                    help="受控全队目标测量中断持续时间[s]")
    ap.add_argument("--belief-source", default=None,
                    choices=["gru", "imm", "kf", "cv", "hybrid"],
                    help="信念源在环验收(方法侧, 须与训练一致, 且 obs_mode 须为 belief): "
                         "复用 train_belief 的信念管理器, 每步 set_belief 注入; "
                         "gru=预测器在环(需--predictor-ckpt), imm/kf/cv=滤波驱动, "
                         "hybrid=IMM先验+GRU残差(需--hybrid-ckpt); "
                         "缺省不启用(行为不变, belief 模式下信念恒 None)")
    ap.add_argument(
        "--belief-variant",
        default=None,
        choices=["point", "belief", "no_covariance", "fixed_confidence",
                 "oracle_confidence"],
                    help="信念注入变体，须与训练一致；managed run 可从同目录 config.yaml "
                         "校验。belief=均值/协方差摘要/置信度；no_covariance="
                         "均值/置信度；fixed_confidence=均值/协方差摘要；"
                         "oracle_confidence=用真实IMM估计误差生成置信度上界；"
                         "point=仅均值")
    ap.add_argument("--predictor-ckpt", default=None,
                    help="gru 信念源的预测器checkpoint路径(须与训练一致)")
    ap.add_argument("--hybrid-ckpt", default=None,
                    help="hybrid信念源的IMM-GRU残差checkpoint路径(须与训练一致)")
    ap.add_argument("--predictor-seq-len", type=int, default=20,
                    help="gru 信念源的观测窗口长度(须与训练一致)")
    ap.add_argument("--lost-bins", default=None,
                    help="失观分档评估协议: 秒切点(逗号分隔, 如 0.5,1,2,3,5), "
                         "按回合最大连续团队失观时长分箱统计; 缺省不启用(行为不变)")
    ap.add_argument("--bins-csv", default=None,
                    help="分箱总表输出CSV路径(须与 --lost-bins 同用)")
    ap.add_argument("--mode-stats-csv", default=None,
                    help="ModeSwitch逐回合统计CSV：切换次数、模式段序列及结局；"
                         "限 evader_type=mode_switch/reactive_mode_switch 的评估配置")
    ap.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="显式允许训练环境与评估环境不同，仅用于预注册的OOD/泛化实验；"
             "方法侧obs_mode、belief_source和belief_variant仍必须与checkpoint一致")
    ap.add_argument(
        "--reacquisition-controller",
        action="store_true",
        help="启用J9 tracker/scout主动重观测动作合成层",
    )
    ap.add_argument(
        "--allow-controller-mismatch",
        action="store_true",
        help="允许在无J9控制器的父checkpoint上做预注册零样本控制器开发诊断",
    )
    ap.add_argument(
        "--scripted-geometric-baseline",
        action="store_true",
        help=("J15 policy-free diagnostic: discard actor actions and keep the "
              "fixed equal-angle/last-observation controller active"),
    )
    ap.add_argument("--reacquisition-start-lost-steps", type=int, default=None)
    ap.add_argument("--reacquisition-slot-radius-m", type=float, default=None)
    ap.add_argument("--reacquisition-scout-blend", type=float, default=None)
    ap.add_argument("--reacquisition-tracker-blend", type=float, default=None)
    ap.add_argument("--reacquisition-blend-growth", type=float, default=None)
    ap.add_argument(
        "--reacquisition-post-hold-steps", type=int, default=None)
    ap.add_argument(
        "--reacquisition-capture-slot-radius-m", type=float, default=None)
    ap.add_argument(
        "--reacquisition-contraction-steps", type=int, default=None)
    ap.add_argument(
        "--reacquisition-visible-tracker-blend", type=float, default=None)
    args = ap.parse_args()
    if args.bins_csv and not args.lost_bins:
        ap.error("--bins-csv 需与 --lost-bins 同用")
    if args.capture_radius is not None and args.capture_radius <= 0:
        ap.error("--capture-radius must be positive")
    if args.min_capture_agents is not None and args.min_capture_agents <= 0:
        ap.error("--min-capture-agents must be positive")
    if args.safety_stats_csv and not args.cbf:
        ap.error("--safety-stats-csv 需与 --cbf 同用")
    if args.safety_version != "legacy" and not args.cbf:
        ap.error("非legacy安全层需显式启用 --cbf")
    if args.belief_source == "gru" and not args.predictor_ckpt:
        ap.error("--belief-source gru 需要 --predictor-ckpt")
    if args.belief_source == "hybrid" and not args.hybrid_ckpt:
        ap.error("--belief-source hybrid 需要 --hybrid-ckpt")
    if args.belief_mechanism_csv and not args.belief_source:
        ap.error("--belief-mechanism-csv要求显式--belief-source")
    if ((args.measurement_blackout_start_s is None)
            != (args.measurement_blackout_duration_s is None)):
        ap.error("measurement blackout开始和持续时间必须同时给出")
    if (
        args.measurement_blackout_start_s is not None
        and (
            args.measurement_blackout_start_s < 0
            or args.measurement_blackout_duration_s <= 0
        )
    ):
        ap.error("measurement blackout要求start>=0且duration>0")

    if args.scripted_geometric_baseline and not args.reacquisition_controller:
        ap.error(
            "--scripted-geometric-baseline requires --reacquisition-controller")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_fields = dict(cfg["env"])
    if args.obs_mode:
        env_fields["obs_mode"] = args.obs_mode
    if args.obstacle_layout is not None:
        env_fields["obstacle_layout"] = args.obstacle_layout
    if args.target_position_noise_std_m is not None:
        env_fields["target_position_noise_std_m"] = (
            args.target_position_noise_std_m)
    if args.sense_range is not None:
        env_fields["sense_range"] = args.sense_range
    if args.visual_occlusion is not None:
        env_fields["visual_occlusion"] = args.visual_occlusion == "true"
    if args.initial_visibility_reference_sense_range is not None:
        env_fields["initial_visibility_reference_sense_range"] = (
            args.initial_visibility_reference_sense_range)
    if args.initial_visibility_reference_visual_occlusion is not None:
        env_fields["initial_visibility_reference_visual_occlusion"] = (
            args.initial_visibility_reference_visual_occlusion == "true")
    if args.team_target_sharing:
        env_fields["team_target_sharing"] = True
    if args.fov_half_deg is not None:
        env_fields["fov_half"] = math.radians(args.fov_half_deg)
    if args.capture_radius is not None:
        env_fields["capture_radius"] = args.capture_radius
    if args.min_capture_agents is not None:
        env_fields["min_capture_agents"] = args.min_capture_agents
    if args.max_gap_deg is not None:
        env_fields["max_gap"] = math.radians(args.max_gap_deg)
    if args.measurement_blackout_start_s is not None:
        dt = float(env_fields.get("dt", EnvConfig.dt))
        env_fields["measurement_blackout_start_step"] = int(round(
            args.measurement_blackout_start_s / dt))
        env_fields["measurement_blackout_duration_steps"] = int(round(
            args.measurement_blackout_duration_s / dt))

    # managed run 的方法侧口径防呆：评估配置只冻结场景/判据，obs_mode、信念源和
    # 注入变体必须与训练 run 一致。此前漏传 --obs-mode last 会静默按 zero 评估。
    run_cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.yaml")
    trained_mode = trained_source = trained_variant = None
    trained_filter_kwargs = {}
    trained_reacquisition_params = {}
    run_train = {}
    if os.path.exists(run_cfg_path):
        with open(run_cfg_path, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f) or {}
        trained_mode = (run_cfg.get("env") or {}).get("obs_mode")
        run_train = run_cfg.get("train") or {}
        trained_source = run_train.get("belief_source")
        trained_variant = run_train.get("belief_variant")
        trained_reacquisition_params = dict(
            run_train.get("reacquisition_controller") or {})
        trained_reacquisition_enabled = bool(
            trained_reacquisition_params
            and trained_reacquisition_params.get("enabled", True))
        if trained_reacquisition_enabled != bool(args.reacquisition_controller) \
                and not args.allow_controller_mismatch:
            expected = "--reacquisition-controller" if trained_reacquisition_enabled else "no controller flag"
            ap.error(
                "J9控制器口径不匹配：checkpoint配置中"
                f"enabled={trained_reacquisition_enabled}，当前评估"
                f"enabled={bool(args.reacquisition_controller)}；应使用{expected}。"
                "仅预注册父checkpoint零样本诊断可显式传"
                "--allow-controller-mismatch。")
        if trained_mode == "belief" and trained_source is None:
            trained_source = "gru"  # train_belief.py 的缺省来源
        requested_mode = env_fields.get("obs_mode", EnvConfig().obs_mode)
        if trained_mode and requested_mode != trained_mode:
            ap.error(
                f"方法口径不匹配：checkpoint 的 config.yaml 为 obs_mode={trained_mode!r}，"
                f"当前评估为 {requested_mode!r}；请显式传 --obs-mode {trained_mode}")
        if trained_mode == "belief":
            if args.belief_source is None:
                ap.error(
                    f"checkpoint 为 belief run（source={trained_source!r}），必须显式传 "
                    f"--belief-source {trained_source}")
            if trained_source and args.belief_source != trained_source:
                ap.error(
                    f"信念源不匹配：checkpoint={trained_source!r}，"
                    f"评估参数={args.belief_source!r}")
            if args.belief_variant is None:
                args.belief_variant = trained_variant or "belief"
            elif trained_variant and args.belief_variant != trained_variant:
                ap.error(
                    f"信念变体不匹配：checkpoint={trained_variant!r}，"
                    f"评估参数={args.belief_variant!r}")
            if args.belief_variant in {
                    "no_covariance", "fixed_confidence",
                    "oracle_confidence"} and \
                    trained_source != "imm":
                ap.error(
                    "J4新增信念变体只预注册用于IMM，"
                    f"当前checkpoint source={trained_source!r}")
            if trained_source in ("cv", "kf", "imm"):
                trained_filter_kwargs = dict(
                    (run_train.get("filter_params") or {}).get(trained_source, {}))
            elif trained_source == "hybrid":
                trained_filter_kwargs = dict(
                    (run_train.get("filter_params") or {}).get("imm", {}))
        mismatches = managed_env_mismatches(run_cfg, env_fields)
        if mismatches:
            detail = "; ".join(
                f"{name}: run={left!r}, eval={right!r}"
                for name, left, right in mismatches
            )
            if not args.allow_env_mismatch:
                ap.error(
                    "环境协议不匹配（仅 obs_mode 属方法侧可覆盖）："
                    f"{detail}。请使用与训练场景对应的固定评估配置；"
                    "若这是预注册的OOD/泛化实验，请显式传 --allow-env-mismatch。")
            print("[OOD协议] 显式允许训练/评估环境差异："
                  f"{detail}", flush=True)

    hidden = args.hidden or run_train.get(
        "hidden", cfg.get("train", {}).get("hidden", 128))
    probe = PursuitLostEnv(EnvConfig(**env_fields))
    actor = build_gaussian_actor(
        probe.obs_dim,
        probe.act_dim,
        hidden=hidden,
        n_agent=probe.n,
        actor_encoder=run_train.get("actor_encoder", "mlp"),
        actor_encoder_kwargs=run_train.get("actor_encoder_kwargs"),
    )
    actor.load_state_dict(torch.load(args.ckpt, map_location="cpu")["actor"])

    ec = probe.cfg   # 含EnvConfig默认值(配置未写的键)
    # 信念源在环(方法侧): 直接复用 train_belief 的管理器与加载逻辑, 勿复制
    mgr_factory = None
    if args.belief_source:
        if ec.obs_mode != "belief":
            ap.error(f"--belief-source 要求 obs_mode=belief(当前 {ec.obs_mode!r}), "
                     "请配 --obs-mode belief 或使用 obs_mode=belief 的评估配置")
        from train_belief import BeliefManager, FilterBeliefManager, load_predictor
        if args.belief_source == "gru":
            predictor, pred_temp = load_predictor(args.predictor_ckpt)

            def mgr_factory(e):
                return BeliefManager(predictor, args.belief_variant or "belief",
                                     e.cfg.arena_size,
                                     e.cfg.lost_cap_steps, args.predictor_seq_len,
                                     pred_temp)
        elif args.belief_source == "hybrid":
            from belief.hybrid import HybridBeliefManager, load_hybrid_checkpoint
            residual_model, hybrid_meta = load_hybrid_checkpoint(args.hybrid_ckpt)
            trained_hybrid_filter = dict(hybrid_meta.get("filter_params") or {})
            if trained_hybrid_filter and trained_hybrid_filter != trained_filter_kwargs:
                ap.error(
                    f"hybrid checkpoint IMM参数{trained_hybrid_filter}与run配置"
                    f"{trained_filter_kwargs}不一致")
            hybrid_seq_len = int(hybrid_meta.get(
                "seq_len", args.predictor_seq_len))
            residual_temperature = float(
                hybrid_meta.get("residual_temperature", 1.0))

            def mgr_factory(e):
                return HybridBeliefManager(
                    residual_model, args.belief_variant or "belief",
                    e.cfg.arena_size, seq_len=hybrid_seq_len,
                    residual_temperature=residual_temperature,
                    filter_kwargs=trained_filter_kwargs)
        else:
            def mgr_factory(e):
                return FilterBeliefManager(args.belief_source,
                                           args.belief_variant or "belief",
                                           e.cfg.arena_size,
                                           filter_kwargs=trained_filter_kwargs,
                                           oracle_confidence_scale_m=float(
                                               run_train.get(
                                                   "oracle_confidence_scale_m",
                                                   5.0)))
    print(f"ckpt={args.ckpt}\neval_config={args.config} "
          f"(rho={ec.capture_radius}, gap={ec.max_gap:.4f}, "
          f"col_term={ec.collision_terminates}, speed_ratio={ec.speed_ratio}, "
          f"layout={ec.obstacle_layout}, "
          f"target_noise={ec.target_position_noise_std_m}m, "
          f"range={ec.sense_range}, fov_half={math.degrees(ec.fov_half):.1f}deg)")
    safety_params = {}
    if args.safety_horizon_s is not None:
        safety_params["horizon_s"] = args.safety_horizon_s
    if args.safety_buffer is not None:
        safety_params["safety_buffer"] = args.safety_buffer
    if args.safety_activation_distance is not None:
        safety_params["activation_distance"] = args.safety_activation_distance
    if args.safety_emergency_ranking is not None:
        safety_params["emergency_ranking"] = args.safety_emergency_ranking
    print(f"cbf={'on' if args.cbf else 'off'} "
          f"safety_version={args.safety_version} "
          f"safety_params={safety_params or 'defaults'}")
    reacquisition_params = {}
    if args.reacquisition_controller:
        reacquisition_params = dict(trained_reacquisition_params)
        reacquisition_params["enabled"] = True
        overrides = {
            "start_lost_steps": args.reacquisition_start_lost_steps,
            "slot_radius_m": args.reacquisition_slot_radius_m,
            "scout_blend": args.reacquisition_scout_blend,
            "tracker_blend": args.reacquisition_tracker_blend,
            "blend_growth": args.reacquisition_blend_growth,
            "post_reacquisition_hold_steps": args.reacquisition_post_hold_steps,
            "capture_slot_radius_m": args.reacquisition_capture_slot_radius_m,
            "contraction_steps": args.reacquisition_contraction_steps,
            "visible_tracker_blend": args.reacquisition_visible_tracker_blend,
        }
        reacquisition_params.update({
            key: value for key, value in overrides.items()
            if value is not None
        })
        if args.scripted_geometric_baseline:
            reacquisition_params.update({
                "always_on_visible": True,
                "scout_blend": 1.0,
                "tracker_blend": 1.0,
                "blend_growth": 0.0,
                "visible_tracker_blend": 1.0,
            })
        # Validate once before launching evaluation episodes.
        build_reacquisition_controller(reacquisition_params)
    print("reacquisition_controller="
          f"{'on' if reacquisition_params else 'off'} "
          f"params={reacquisition_params or 'none'}")
    print("nominal_policy="
          f"{'zeroed_for_scripted_geometry' if args.scripted_geometric_baseline else 'learned_actor'}")
    if args.belief_source:
        if args.belief_source == "gru":
            extra = (f" predictor={args.predictor_ckpt} "
                     f"seq_len={args.predictor_seq_len}")
        elif args.belief_source == "hybrid":
            extra = (f" hybrid={args.hybrid_ckpt} "
                     f"params={trained_filter_kwargs or 'defaults'}")
        else:
            extra = f" params={trained_filter_kwargs or 'defaults'}"
        print(f"belief-source={args.belief_source} "
              f"(在环注入, variant={args.belief_variant or 'belief'}){extra}")
    edges_s = edges_steps = None
    if args.lost_bins:
        edges_s, edges_steps = parse_lost_bins(args.lost_bins, ec.dt)
        print(f"lost-bins(s)={args.lost_bins} dt={ec.dt} "
              f"steps_edges={edges_steps[:-1] + ['inf']}")
    if args.mode_stats_csv and ec.evader_type not in {
            "mode_switch", "reactive_mode_switch"}:
        ap.error("--mode-stats-csv 仅适用于 evader_type=mode_switch 的评估配置")
    if args.visibility_stats_csv and not ec.visibility_diagnostics:
        ap.error(
            "--visibility-stats-csv要求评估配置visibility_diagnostics=true")
    succ, seed_accs, mode_records, safety_records = [], [], [], []
    visibility_records = []
    belief_mechanism_records = []
    reward_records = []
    capture_sensitivity_records = []
    step_telemetry_records = []
    for s in range(args.eval_seeds):
        base = args.eval_seed_base + s * 10_000
        r = run_block(actor, env_fields, args.episodes, base, use_cbf=args.cbf,
                      collect=(args.lost_bins is not None or
                               args.mode_stats_csv is not None or
                               args.visibility_stats_csv is not None or
                               args.belief_mechanism_csv is not None or
                               args.reward_stats_csv is not None or
                               args.capture_sensitivity_csv is not None),
                      mgr_factory=mgr_factory,
                      safety_version=args.safety_version,
                      safety_params=safety_params,
                      reacquisition_params=reacquisition_params,
                      zero_policy_actions=args.scripted_geometric_baseline,
                      collect_steps=args.step_telemetry_csv is not None)
        succ.append(r["success"])
        print(f"[eval-seed {s} base={base}] success={r['success']:.3f} "
              f"collision={r['collision']:.3f} timeout={r['timeout']:.3f} "
              f"long_lost={r['long_lost']:.3f} mean_len={r['mean_len']:.1f}",
              flush=True)
        if args.lost_bins:
            seed_accs.append(accumulate_bins(r["episodes"], edges_steps))
        if args.mode_stats_csv:
            for record in r["episodes"]:
                mode_records.append({"eval_seed": s, **record})
        if args.safety_stats_csv:
            for record in r["safety_episodes"]:
                safety_records.append({"eval_seed": s, **record})
        if args.visibility_stats_csv:
            for record in r["episodes"]:
                visibility_records.append({"eval_seed": s, **record})
        if args.belief_mechanism_csv:
            for record in r["episodes"]:
                belief_mechanism_records.append(
                    {"eval_seed": s, **record})
        if args.reward_stats_csv:
            for record in r["episodes"]:
                reward_records.append({"eval_seed": s, **record})
        if args.capture_sensitivity_csv:
            for record in r["episodes"]:
                capture_sensitivity_records.append(
                    {"eval_seed": s, **record})
        if args.step_telemetry_csv:
            for record in r["steps"]:
                step_telemetry_records.append(
                    {"eval_seed": s, **record})
    print(f"\nsuccess mean±std: {np.mean(succ):.3f} ± {np.std(succ):.3f} "
          f"({args.eval_seeds}评估种子 x {args.episodes}回合)")
    if args.lost_bins:
        rows = bin_table_rows(merge_bins(seed_accs), edges_s)
        print_bin_table(rows, args.eval_seeds, args.episodes)
        if args.bins_csv:
            write_bins_csv(args.bins_csv, rows)
            print(f"bins_csv -> {args.bins_csv}")
    if args.mode_stats_csv:
        write_mode_stats_csv(args.mode_stats_csv, mode_records)
        switches = np.asarray(
            [r["evader_mode_switches"] for r in mode_records], dtype=float)
        print(f"mode-switches mean±std: {switches.mean():.3f} ± "
              f"{switches.std():.3f} ({len(mode_records)}回合)")
        print(f"mode_stats_csv -> {args.mode_stats_csv}")
    if args.safety_stats_csv:
        write_safety_stats_csv(args.safety_stats_csv, safety_records)
        print(f"safety_stats_csv -> {args.safety_stats_csv}")
    if args.visibility_stats_csv:
        write_visibility_stats_csv(
            args.visibility_stats_csv, visibility_records)
        total_lost = sum(
            record["loss_steps_pure_range"]
            + record["loss_steps_pure_fov"]
            + record["loss_steps_pure_occlusion"]
            + record["loss_steps_mixed"]
            for record in visibility_records
        )
        pure_occlusion = sum(
            record["loss_steps_pure_occlusion"]
            for record in visibility_records
        )
        print(
            "pure-occlusion team-loss step share: "
            f"{pure_occlusion / max(total_lost, 1):.3f} "
            f"({pure_occlusion}/{total_lost})")
        print(f"visibility_stats_csv -> {args.visibility_stats_csv}")
    if args.belief_mechanism_csv:
        write_belief_mechanism_csv(
            args.belief_mechanism_csv, belief_mechanism_records)
        coverage = np.asarray([
            record["suspected_region_coverage95_rate"]
            for record in belief_mechanism_records
        ], float)
        duplicate = np.asarray([
            record["duplicate_coverage_rate"]
            for record in belief_mechanism_records
        ], float)
        print(
            "belief mechanism: "
            f"coverage95={np.nanmean(coverage):.3f}, "
            f"duplicate={np.nanmean(duplicate):.3f}")
        print(f"belief_mechanism_csv -> {args.belief_mechanism_csv}")
    if args.reward_stats_csv:
        write_reward_stats_csv(args.reward_stats_csv, reward_records)
        reward_total = np.asarray([
            record["reward_sum"] for record in reward_records
        ], float)
        switches = np.asarray([
            record["reward_reference_switches"] for record in reward_records
        ], float)
        print(
            "reward audit: "
            f"return={reward_total.mean():.3f}, "
            f"reference_switches={switches.mean():.3f}/episode")
        print(f"reward_stats_csv -> {args.reward_stats_csv}")
    if args.capture_sensitivity_csv:
        write_capture_sensitivity_csv(
            args.capture_sensitivity_csv, capture_sensitivity_records)
        print("capture sensitivity success rates:")
        for name in CAPTURE_SENSITIVITY_FIELDS:
            values = np.asarray([
                int(record[name]) >= 0
                for record in capture_sensitivity_records
            ], float)
            print(f"  {name.removesuffix('_step')}={values.mean():.3f}")
        print(
            "capture_sensitivity_csv -> "
            f"{args.capture_sensitivity_csv}")
    if args.step_telemetry_csv:
        write_step_telemetry_csv(
            args.step_telemetry_csv, step_telemetry_records)
        print(
            f"step_telemetry_csv -> {args.step_telemetry_csv} "
            f"({len(step_telemetry_records)} rows)")


if __name__ == "__main__":
    main()
