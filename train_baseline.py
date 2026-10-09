# -*- coding: utf-8 -*-
"""MAPPO 基线训练入口(里程碑 M1: C0 档无失观场景收敛).

用法(本机 rl_env):
    conda activate rl_env
    python train_baseline.py --config configs/c0_oracle.yaml
    python train_baseline.py --config configs/c1_zero.yaml    # 失观清零基线

输出: runs/<tag>/ 下保存 config 快照、训练曲线 csv、最优 checkpoint.
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
from safety.cbf_qp import CBFSafetyLayer
from safety.factory import build_safety_layer
from utils.reacquisition import build_reacquisition_controller


def _agent_vel(env, j):
    import math as _m
    return env.p_v[j] * np.array([_m.cos(env.p_psi[j]), _m.sin(env.p_psi[j])])


def apply_safety(env, act, layer, return_diagnostics=False):
    """CBF安全层作为环境级动作滤波器: 策略动作(归一化)逐机投影为安全动作.
    buffer仍存策略原始动作(safety wrapper视作环境动力学的一部分)."""
    if hasattr(layer, "project_joint"):
        out, diagnostics = layer.project_joint(env, act)
        return (out, diagnostics) if return_diagnostics else out
    out = act.copy()
    corrections = []
    correction_norms = []
    obstacles = [(env.obs_pos[k], env.obs_r[k], env.obs_vel[k])
                 for k in range(len(env.obs_pos))]
    for i in range(env.n):
        neighbors = [(env.p_pos[j], _agent_vel(env, j)) for j in range(env.n) if j != i]
        out[i], corrected = layer.project(
            act[i],
            env.p_pos[i],
            env.p_psi[i],
            env.p_v[i],
            neighbors,
            obstacles,
        )
        corrections.append(bool(corrected))
        correction_norms.append(float(np.linalg.norm(out[i] - act[i])))
    diagnostics = {
        "safety_version": "legacy",
        "corrected": any(corrections),
        "corrected_agents": sum(corrections),
        "correction_norm": float(np.linalg.norm(out - act)),
        "max_agent_correction_norm": max(correction_norms, default=0.0),
    }
    return (out, diagnostics) if return_diagnostics else out


def make_envs(env_cfg: EnvConfig, n_env: int, base_seed: int):
    envs = []
    for k in range(n_env):
        cfg_k = EnvConfig(**{**asdict(env_cfg), "seed": base_seed + k})
        envs.append(PursuitLostEnv(cfg_k))
    return envs


def evaluate(
    agent,
    env_cfg,
    episodes=20,
    seed=10_000,
    device="cpu",
    use_cbf=False,
    safety_version="legacy",
    safety_params=None,
    reacquisition_params=None,
):
    stats = dict(success=0, collision=0, timeout=0, long_lost=0, ep_len=[], recap=0, lost_ev=0)
    layer = (
        build_safety_layer(env_cfg, safety_version, safety_params)
        if use_cbf else None
    )
    for e in range(episodes):
        env = PursuitLostEnv(EnvConfig(**{**asdict(env_cfg), "seed": seed + e}))
        obs, state = env.reset()
        reacquisition = build_reacquisition_controller(reacquisition_params)
        done, t = False, 0
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(o)
                act = torch.tanh(mean)[0].cpu().numpy()
            if reacquisition is not None:
                act, _ = reacquisition.apply(env, act)
            if layer is not None:
                act = apply_safety(env, act, layer)
            obs, state, _, done, info = env.step(act)
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
    ap.add_argument("--config", default="configs/c0_oracle.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=None, help="覆盖配置中的训练迭代数")
    ap.add_argument("--resume", default=None, help="从checkpoint继续训练(课程分阶段用)")
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置中的训练种子(多种子纪律)")
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
        # eval_env is an override layer just like env/train.  A shallow
        # setdefault drops recipe safety fields whenever an experiment adds a
        # single evaluation override (the J9 k_rec-only config exposed this).
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
    # 评估环境固定为最终判据: env终值 + anneal终点 + eval_env覆盖, 训练期永不变动
    eval_fields = dict(cfg["env"])
    if tr.get("anneal"):
        for f_name, (v0, v1) in tr["anneal"]["fields"].items():
            eval_fields[f_name] = float(v1)
    eval_fields.update(cfg.get("eval_env", {}) or {})
    eval_cfg = EnvConfig(**eval_fields)
    tag = (cfg.get("tag", "run") + f"_s{tr['seed']}" + time.strftime("_%m%d_%H%M"))
    out = os.path.join("runs", tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    # 登记: (recipe版本, 判据参数, 种子, 承接run)
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
    variants = list(episode_mix.get("variants") or [])
    if variants and (easy_fields or "easy_probability" in episode_mix):
        raise ValueError(
            "train.episode_mix.variants cannot be combined with legacy easy_fields")
    if not 0.0 <= easy_probability <= 1.0:
        raise ValueError("train.episode_mix.easy_probability must be in [0, 1]")
    parsed_variants = []
    variant_field_names = set()
    probability_sum = 0.0
    for index, raw_variant in enumerate(variants):
        probability = float(raw_variant.get("probability", 0.0))
        fields = dict(raw_variant.get("fields") or {})
        if probability <= 0.0 or probability > 1.0:
            raise ValueError(
                f"train.episode_mix.variants[{index}].probability must be in (0, 1]")
        if not fields:
            raise ValueError(
                f"train.episode_mix.variants[{index}].fields must be non-empty")
        unknown = sorted(name for name in fields if not hasattr(env_cfg, name))
        if unknown:
            raise ValueError(
                f"train.episode_mix.variants[{index}] unknown fields: {unknown}")
        probability_sum += probability
        variant_field_names.update(fields)
        parsed_variants.append((probability, fields))
    if probability_sum > 1.0 + 1e-12:
        raise ValueError("train.episode_mix.variants probabilities must sum to <= 1")
    hard_fields = {
        name: getattr(env_cfg, name)
        for name in set(easy_fields) | variant_field_names
    }
    mix_rngs = [
        np.random.default_rng(int(tr["seed"]) * 100_000 + k)
        for k in range(n_env)
    ]

    def reset_training_env(index):
        env = envs[index]
        for name, hard_value in hard_fields.items():
            setattr(env.cfg, name, hard_value)
        if parsed_variants:
            draw = mix_rngs[index].random()
            cumulative = 0.0
            selected = None
            for probability, fields in parsed_variants:
                cumulative += probability
                if draw < cumulative:
                    selected = fields
                    break
            for name, value in (selected or {}).items():
                setattr(env.cfg, name, value)
        else:
            use_easy = (
                bool(easy_fields)
                and mix_rngs[index].random() < easy_probability
            )
            for name, hard_value in hard_fields.items():
                setattr(
                    env.cfg, name,
                    easy_fields[name] if use_easy else hard_value)
        return env.reset()

    obs = np.stack([reset_training_env(k)[0] for k in range(n_env)])
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
    # 每迭代对 EnvConfig 浮点字段线性插值(含eval用的env_cfg), 平滑收紧判据
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
                    ob, st = reset_training_env(k)
                    if reacquisition_controllers[k] is not None:
                        reacquisition_controllers[k].reset()
                    dn[k] = 1.0
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
