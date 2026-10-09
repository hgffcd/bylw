# -*- coding: utf-8 -*-
"""回合可视化渲染(GIF), 里程碑 M1 验收工具.

移植自 D:\\p_g_weibu\\visualize.py (MRA-RLEC 复现的 Matplotlib 动画), 适配本项目
PursuitLostEnv 的 unicycle 状态、FOV 视场、失观标志与信念注入接口.

用法(本机 rl_env):
    conda activate rl_env
    python render.py --config configs/c0_oracle.yaml --steps 80          # 随机策略冒烟
    python render.py --config configs/c0_oracle.yaml --checkpoint runs/<tag>/best.pt

画面元素:
- 场地边界 + 网格; 静态障碍(灰圆) / 动态障碍(橙圆, 随帧运动);
- 追捕者: 三角形指向航向 psi + FOV 扇形楔(探测距离截断) + 捕获半径虚线圈(rho_c);
- 目标: 团队可见 = 实心红圆, 失观 = 空心虚线圆; 追捕者/目标轨迹尾迹;
- HUD: step / t_lost / 可见机数 / 回合累计回报 / 终局结果;
- obs_mode=belief 且外部注入信念时, 画信念均值(x) + 2σ 协方差椭圆(本脚本随机/
  checkpoint 回放不注入信念, 该层自动跳过).

约定: checkpoint 为 train_baseline.py 保存格式 dict(actor=..., critic=...),
actor 结构 = algos/networks.py 的 GaussianActor(obs_dim, act_dim, hidden).
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict

import numpy as np
import yaml

from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv

# 与 p_g_weibu/visualize.py 一致的追捕者配色
PURSUER_COLORS = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#17becf", "#8c564b", "#e377c2"]


# ---------------- 策略 ----------------
def make_policy(args, env: PursuitLostEnv, cfg: dict):
    """返回 policy(obs) -> (n,2) 动作. 无 checkpoint 用随机策略(不依赖 torch)."""
    if not args.checkpoint:
        rng = np.random.default_rng(args.seed if args.seed is not None else 0)

        def random_policy(obs):
            return rng.uniform(-1.0, 1.0, (env.n, env.act_dim))

        return random_policy, "random"

    import torch
    from algos.networks import GaussianActor

    hidden = int(cfg.get("train", {}).get("hidden", 128))
    actor = GaussianActor(env.obs_dim, env.act_dim, hidden)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    def ckpt_policy(obs):
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = actor(o)                      # 与 train_baseline.evaluate 一致: 取确定性均值
            return torch.tanh(mean).numpy()

    return ckpt_policy, f"checkpoint({os.path.basename(args.checkpoint)})"


# ---------------- 回合采样 ----------------
def rollout(env: PursuitLostEnv, policy, max_steps: int,
            belief_manager=None, safety_layer=None,
            reacquisition_controller=None):
    """执行一回合并记录逐帧快照.

    belief_manager、reacquisition_controller与safety_layer缺省均关闭，保持
    原随机/普通checkpoint回放行为；正式论文回放传入时，动作时序严格对齐
    eval_policy.run_block：策略 -> 重观测控制器 -> 安全层 -> 环境。
    """
    obs, _ = env.reset()
    if belief_manager is not None:
        obs = belief_manager.reset(env)
    if reacquisition_controller is not None:
        reacquisition_controller.reset()
    frames = []

    def snapshot(ep_ret: float):
        b = None
        if env.belief is not None:                  # belief 模式外部注入时才存在
            b = dict(mu=np.array(env.belief["mu"], float),
                     sigma=np.array(env.belief["sigma"], float),
                     conf=float(env.belief["conf"]))
        frames.append(dict(
            p_pos=env.p_pos.copy(), p_psi=env.p_psi.copy(),
            t_pos=env.t_pos.copy(),
            obs_pos=env.obs_pos.copy(),
            visible_each=env.visible_each.copy(),
            t_lost=int(env.t_lost), ep_ret=float(ep_ret), belief=b,
            evader_mode=(
                int(env._mode_evader.mode)
                if getattr(env, "_mode_evader", None) is not None else None),
        ))

    ep_ret = 0.0
    snapshot(ep_ret)
    info = dict(success=False, collision=False, timeout=False, long_lost=False)
    for _ in range(max_steps):
        act = policy(obs)
        if reacquisition_controller is not None:
            act, _ = reacquisition_controller.apply(env, act)
        if safety_layer is not None:
            from train_baseline import apply_safety
            act = apply_safety(env, act, safety_layer)
        obs, _, rew, done, info = env.step(act)
        if belief_manager is not None:
            obs = belief_manager.inject(env)
        ep_ret += float(rew[0])
        snapshot(ep_ret)
        if done:
            break
    if info["success"]:
        result = "SUCCESS"
    elif info["collision"]:
        result = "COLLISION"
    elif info["long_lost"]:
        result = "LONG_LOST"
    elif info["timeout"]:
        result = "TIMEOUT"
    else:
        result = "RUNNING"
    return frames, result, info


# ---------------- 渲染 ----------------
def save_gif(frames, result, env: PursuitLostEnv, out_path: str, fps: int, stride: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, Wedge

    c = env.cfg
    L = c.arena_size
    n = env.n
    # 帧抽样(保留末帧, 控制 GIF 体积)
    idx = list(range(0, len(frames), max(stride, 1)))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    frames = [frames[i] for i in idx]

    # 静/动态障碍区分: 采样时动态障碍在末尾且速度非零
    dyn_mask = np.linalg.norm(env.obs_vel, axis=1) > 1e-9 if len(env.obs_pos) else np.zeros(0, bool)
    obs_r = env.obs_r

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    pad = 0.03 * L
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, L + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, lw=1.5, color="black"))  # 场地边界

    # 障碍 patch(动态障碍逐帧移动)
    obstacle_patches = []
    for j in range(len(obs_r)):
        color = "#ff7f0e" if dyn_mask[j] else "0.55"
        p = Circle(tuple(frames[0]["obs_pos"][j]), obs_r[j], color=color,
                   alpha=0.75 if dyn_mask[j] else 0.6, lw=0, zorder=2)
        ax.add_patch(p)
        obstacle_patches.append(p)

    # FOV 楔: 半径截断到场地内; 全向且全域感知(如 C0 档)时不画
    draw_fov = not (c.fov_half >= math.pi - 1e-3 and c.sense_range >= L)
    fov_r = min(c.sense_range, L)

    def tri_verts(pos, psi):
        """朝向 psi 的等腰三角形顶点(机体图标)."""
        tip = pos + 2.2 * np.array([math.cos(psi), math.sin(psi)])
        left = pos + 1.5 * np.array([math.cos(psi + 2.5), math.sin(psi + 2.5)])
        right = pos + 1.5 * np.array([math.cos(psi - 2.5), math.sin(psi - 2.5)])
        return np.stack([tip, left, right])

    tri_patches, fov_patches, cap_patches, trail_lines = [], [], [], []
    f0 = frames[0]
    for i in range(n):
        col = PURSUER_COLORS[i % len(PURSUER_COLORS)]
        tri = Polygon(tri_verts(f0["p_pos"][i], f0["p_psi"][i]), closed=True,
                      color=col, zorder=5)
        ax.add_patch(tri)
        tri_patches.append(tri)
        if draw_fov:
            w = Wedge(tuple(f0["p_pos"][i]), fov_r,
                      math.degrees(f0["p_psi"][i] - c.fov_half),
                      math.degrees(f0["p_psi"][i] + c.fov_half),
                      facecolor=col, alpha=0.08, lw=0, zorder=1)
            ax.add_patch(w)
            fov_patches.append(w)
        cap = Circle(tuple(f0["p_pos"][i]), c.capture_radius, fill=False,
                     ls="--", lw=1.0, color=col, alpha=0.5, zorder=3)
        ax.add_patch(cap)
        cap_patches.append(cap)
        line, = ax.plot([], [], color=col, lw=1.1, alpha=0.7, zorder=4)
        trail_lines.append(line)

    # 目标: 可见=实心红, 失观=空心虚线
    target_patch = Circle(tuple(f0["t_pos"]), 1.2, facecolor="#d62728",
                          edgecolor="#d62728", lw=1.5, zorder=6)
    ax.add_patch(target_patch)
    target_trail, = ax.plot([], [], color="#d62728", lw=1.0, ls=":", alpha=0.6, zorder=4)

    # 信念层(均值 x + 2σ 椭圆), 无信念数据时保持隐藏
    belief_mean, = ax.plot([], [], marker="x", ms=9, mew=2, color="#e377c2",
                           ls="none", zorder=6)
    belief_ell = Ellipse((0, 0), 0.0, 0.0, angle=0.0, fill=False, ls="-.",
                         lw=1.3, color="#e377c2", alpha=0.9, zorder=5, visible=False)
    ax.add_patch(belief_ell)

    hud = ax.text(0.02, 0.985, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=9, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.set_title(f"pursuit-lost episode  (n={n}, rho_c={c.capture_radius:g} m, "
                 f"FOV={math.degrees(c.fov_half):.0f} deg, Rs={c.sense_range:g} m)",
                 fontsize=10)

    p_hist = np.stack([f["p_pos"] for f in frames])       # (T, n, 2)
    t_hist = np.stack([f["t_pos"] for f in frames])       # (T, 2)

    def belief_ellipse_params(sigma_feat):
        """归一化[Sxx,Syy,Sxy] -> 米制(宽,高,角度), 2σ椭圆."""
        s = np.asarray(sigma_feat, float).ravel()
        if s.size < 3:
            var = max(float(s.mean()) * L ** 2, 1e-6)
            return 4 * math.sqrt(var), 4 * math.sqrt(var), 0.0
        cov = np.array([[s[0], s[2]], [s[2], s[1]]], float) * L ** 2
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-9, None)
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        return 4 * math.sqrt(vals[1]), 4 * math.sqrt(vals[0]), ang

    def update(k):
        f = frames[k]
        for j, p in enumerate(obstacle_patches):
            p.center = tuple(f["obs_pos"][j])
        for i in range(n):
            tri_patches[i].set_xy(tri_verts(f["p_pos"][i], f["p_psi"][i]))
            cap_patches[i].center = tuple(f["p_pos"][i])
            if draw_fov:
                fov_patches[i].set_center(tuple(f["p_pos"][i]))
                fov_patches[i].set_theta1(math.degrees(f["p_psi"][i] - c.fov_half))
                fov_patches[i].set_theta2(math.degrees(f["p_psi"][i] + c.fov_half))
                fov_patches[i].set_alpha(0.16 if f["visible_each"][i] else 0.06)
            trail_lines[i].set_data(p_hist[: k + 1, i, 0], p_hist[: k + 1, i, 1])
        team_visible = bool(f["visible_each"].any())
        target_patch.center = tuple(f["t_pos"])
        target_patch.set_facecolor("#d62728" if team_visible else "none")
        target_patch.set_linestyle("-" if team_visible else "--")
        target_trail.set_data(t_hist[: k + 1, 0], t_hist[: k + 1, 1])
        if f["belief"] is not None:
            mu = f["belief"]["mu"]
            belief_mean.set_data([mu[0]], [mu[1]])
            w, h, ang = belief_ellipse_params(f["belief"]["sigma"])
            belief_ell.set_center((mu[0], mu[1]))
            belief_ell.width, belief_ell.height, belief_ell.angle = w, h, ang
            belief_ell.set_visible(True)
        else:
            belief_mean.set_data([], [])
            belief_ell.set_visible(False)
        res = result if k == len(frames) - 1 else "..."
        mode_names = ("CV", "CT", "STOP")
        mode = (
            mode_names[f["evader_mode"]]
            if f["evader_mode"] is not None else
            ("INIT" if c.evader_type == "mode_switch" else "APF"))
        hud.set_text(f"step={idx[k]:4d}  t_lost={f['t_lost']:3d}  "
                     f"visible={int(f['visible_each'].sum())}/{n}  mode={mode}\n"
                     f"return={f['ep_ret']:8.2f}  result={res}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=max(1, int(1000 / fps)), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="PursuitLostEnv 回合 GIF 渲染")
    ap.add_argument("--config", required=True, help="课程档位配置 yaml")
    ap.add_argument("--checkpoint", default=None, help="train_baseline 保存的 .pt(缺省用随机策略)")
    ap.add_argument("--steps", type=int, default=500, help="最大回放步数")
    ap.add_argument("--out", default=None, help="输出 GIF 路径, 默认 runs/render_<tag>.gif")
    ap.add_argument("--seed", type=int, default=None, help="环境随机种子")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1, help="每隔几步取一帧(控制 GIF 体积)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_kwargs = dict(cfg["env"])
    if args.seed is not None:
        env_kwargs["seed"] = args.seed
    env_cfg = EnvConfig(**env_kwargs)
    env = PursuitLostEnv(env_cfg)

    policy, policy_name = make_policy(args, env, cfg)
    frames, result, info = rollout(env, policy, args.steps)

    out = args.out or os.path.join("runs", f"render_{cfg.get('tag', 'run')}.gif")
    save_gif(frames, result, env, out, fps=args.fps, stride=args.stride)
    st = info["stats"] if "stats" in info else {}
    print(f"saved={out} size={os.path.getsize(out) / 1024:.1f}KB "
          f"policy={policy_name} steps={len(frames) - 1} result={result} "
          f"t_lost_max={st.get('max_lost', 0)} recaptures={st.get('recaptures', 0)} "
          f"collisions={st.get('collisions', 0)}")


if __name__ == "__main__":
    main()
