# -*- coding: utf-8 -*-
"""采集目标轨迹数据集(预测器训练用, 对应三阶段训练流程步骤1).

用法: python scripts/collect_trajectories.py --episodes 500 --out data/traj.npz
策略: 追捕方用APF式简单追击+随机混合, 保证数据覆盖追击压力下的目标机动与失观片段.
输出: 每回合的目标绝对轨迹、速度、团队可见性标志序列.

新增(2026-07-27, M3刷新代码准备, 硬约束6):
  --policy-ckpt ckpt  加载训练好的MAPPO actor checkpoint(格式dict含'actor'键,
                      algos/networks.py::GaussianActor, 确定性tanh(mean)推理),
                      与APF混合策略按比例混合采集; RL策略回合动作全程过CBF安全层
                      (与训练配方recipe_v2一致, train_baseline.apply_safety).
  --policy-mix p      RL策略回合占比(默认0.5, 其余回合仍用APF混合策略).
  --policy-hidden h   actor隐层宽度(默认128, 须与训练配置一致).
不带新参数时行为与旧版逐位一致(不额外消耗随机数流); 输出npz格式不变.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv  # noqa: E402


def apf_chase_action(env: PursuitLostEnv, i: int, rng) -> np.ndarray:
    """简单追击引导: 朝目标(或最后观测点)转向并加速, 叠加噪声."""
    ref = env.t_pos if env.visible_each.any() else env.last_seen_pos
    d = ref - env.p_pos[i]
    bearing = np.arctan2(d[1], d[0])
    err = np.arctan2(np.sin(bearing - env.p_psi[i]), np.cos(bearing - env.p_psi[i]))
    w = np.clip(err / env.cfg.w_max, -1, 1)
    a = 1.0 if abs(err) < 1.0 else 0.3
    act = np.array([a, w]) + rng.normal(0, 0.15, 2)
    return np.clip(act, -1, 1)


def load_policy(ckpt_path: str, hidden: int, probe_cfg: EnvConfig):
    """加载MAPPO actor(checkpoint格式: dict含'actor'键的state_dict), 冻结为确定性策略."""
    import torch
    from algos.networks import GaussianActor
    probe = PursuitLostEnv(probe_cfg)
    actor = GaussianActor(probe.obs_dim, probe.act_dim, hidden)
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu")["actor"])
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="data/traj.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fov-deg", type=float, default=60)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--policy-ckpt", default=None,
                    help="MAPPO actor checkpoint(dict含'actor'键); 提供后按--policy-mix混采")
    ap.add_argument("--policy-mix", type=float, default=0.5,
                    help="RL策略回合占比(仅在--policy-ckpt提供时生效)")
    ap.add_argument("--policy-hidden", type=int, default=128,
                    help="actor隐层宽度(须与训练一致, recipe_v2=128)")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="RL采集推理的CPU线程数；并行采集建议设为1")
    ap.add_argument("--speed-ratio", type=float, default=None,
                    help="覆盖目标速度比(默认用EnvConfig默认值0.9; C1档=0.6)")
    ap.add_argument("--n-dyn-obs", type=int, default=None,
                    help="覆盖动态障碍数(默认EnvConfig默认2; C1档=0)")
    ap.add_argument("--obs-mode", choices=["zero", "last", "belief"], default="zero",
                    help="环境观测模式；加载策略时必须与该策略训练口径一致")
    ap.add_argument("--evader-type", choices=["apf", "mode_switch"], default=None,
                    help="目标机动模型；省略时保持环境历史默认APF")
    ap.add_argument("--dataset-role", default=None,
                    help="论文数据集角色标签，如APF_ID或ModeSwitch_OOD；提供后写入元数据")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖回合上限；省略时保持环境默认")
    ap.add_argument("--sense-range", type=float, default=None,
                    help="覆盖感知距离；省略时保持环境默认")
    ap.add_argument("--safety-version", choices=["legacy", "dynamic_v2"],
                    default="legacy",
                    help="RL采集回合使用的安全层版本")
    ap.add_argument("--safety-params-json", default=None,
                    help="安全层参数JSON；例如dynamic_v2的horizon_s等")
    ap.add_argument("--safety-params-file", default=None,
                    help="安全层参数JSON文件；与--safety-params-json二选一")
    args = ap.parse_args()
    if args.safety_params_json and args.safety_params_file:
        raise SystemExit(
            "--safety-params-json与--safety-params-file只能提供一个")
    rng = np.random.default_rng(args.seed)

    def make_cfg(ep):
        extra = {}
        if args.speed_ratio is not None:
            extra["speed_ratio"] = args.speed_ratio
        if args.n_dyn_obs is not None:
            extra["n_dynamic_obs"] = args.n_dyn_obs
        if args.evader_type is not None:
            extra["evader_type"] = args.evader_type
        if args.max_steps is not None:
            extra["max_steps"] = args.max_steps
        if args.sense_range is not None:
            extra["sense_range"] = args.sense_range
        return EnvConfig(seed=args.seed * 100000 + ep,
                         fov_half=np.radians(args.fov_deg),
                         n_static_obs=args.n_obs, obs_mode=args.obs_mode, **extra)

    actor = None
    if args.policy_ckpt:
        # 惰性导入torch链, 保证不带新参数时脚本行为(含依赖面)与旧版一致
        import torch
        from safety.factory import build_safety_layer
        from train_baseline import apply_safety
        if args.torch_threads is not None:
            if args.torch_threads < 1:
                raise SystemExit("--torch-threads必须>=1")
            torch.set_num_threads(args.torch_threads)
            torch.set_num_interop_threads(1)
        if args.safety_params_file:
            with open(args.safety_params_file, encoding="utf-8") as stream:
                safety_params = json.load(stream)
        else:
            safety_params = (
                json.loads(args.safety_params_json)
                if args.safety_params_json else {})
        actor = load_policy(args.policy_ckpt, args.policy_hidden, make_cfg(0))
        print(f"RL策略已加载: {args.policy_ckpt} (mix={args.policy_mix:.2f}, "
              f"hidden={args.policy_hidden}, "
              f"RL回合动作过{args.safety_version}安全层)")

    trajs, vels, vis = [], [], []
    episode_policy_labels = []
    n_rl = 0
    for ep in range(args.episodes):
        env = PursuitLostEnv(make_cfg(ep))
        obs, _ = env.reset()
        # 仅在加载了RL策略时才消耗混合抽签随机数(保持无新参数时随机数流不变)
        use_rl = actor is not None and (float(rng.random()) < args.policy_mix)
        episode_policy_labels.append("rl_cbf" if use_rl else "apf_chase")
        layer = (
            build_safety_layer(
                env.cfg, args.safety_version, safety_params)
            if use_rl else None
        )
        n_rl += int(use_rl)
        tp, tv, tf = [], [], []
        done = False
        while not done:
            if use_rl:
                with torch.no_grad():
                    mu, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
                    acts = torch.tanh(mu).numpy()          # 确定性策略
                acts = apply_safety(env, acts, layer)      # CBF安全投影(同recipe_v2部署口径)
            else:
                acts = np.stack([apf_chase_action(env, i, rng) for i in range(env.n)])
            obs, _, _, done, info = env.step(acts)
            tp.append(env.t_pos.copy())
            tv.append(env.t_vel.copy())
            tf.append(1.0 if info["visible"] else 0.0)
        trajs.append(np.array(tp))
        vels.append(np.array(tv))
        vis.append(np.array(tf))
        if (ep + 1) % 50 == 0:
            lost_ratio = float(np.mean([1 - v.mean() for v in vis]))
            extra = f"  RL回合占比={n_rl/(ep+1):.2%}" if actor is not None else ""
            print(f"ep {ep+1}/{args.episodes}  平均失观占比={lost_ratio:.2%}{extra}")
    if actor is not None:
        print(f"混合采集完成: RL回合 {n_rl}/{args.episodes} ({n_rl/args.episodes:.2%}), "
              f"其余为APF混合策略回合")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "traj": np.array(trajs, dtype=object),
        "vel": np.array(vels, dtype=object),
        "vis": np.array(vis, dtype=object),
    }
    if args.dataset_role is not None or args.evader_type is not None:
        resolved_cfg = make_cfg(0)
        output["episode_policy"] = np.array(episode_policy_labels)
        output["metadata_json"] = np.array(json.dumps({
            "schema_version": 1,
            "dataset_role": args.dataset_role,
            "evader_type": resolved_cfg.evader_type,
            "collector": "scripts/collect_trajectories.py",
            "seed": args.seed,
            "episodes": args.episodes,
            "fov_deg": args.fov_deg,
            "n_static_obs": args.n_obs,
            "n_dynamic_obs": resolved_cfg.n_dynamic_obs,
            "speed_ratio": resolved_cfg.speed_ratio,
            "obs_mode": args.obs_mode,
            "policy_checkpoint": args.policy_ckpt,
            "policy_mix": args.policy_mix if args.policy_ckpt else 0.0,
            "safety_version": (
                args.safety_version if args.policy_ckpt else None),
            "safety_params": safety_params if args.policy_ckpt else {},
            "sense_range": resolved_cfg.sense_range,
            "max_steps": resolved_cfg.max_steps,
        }, ensure_ascii=False))
    np.savez_compressed(args.out, **output)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
