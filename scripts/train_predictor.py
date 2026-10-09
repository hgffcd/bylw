# -*- coding: utf-8 -*-
"""GRU概率预测器训练(三阶段流程步骤2, 里程碑M3).

用法: python scripts/train_predictor.py --data data/traj.npz --epochs 50
评估: 聚合 ADE / NLL / PICP@95 + per-horizon(k=1..K) 报表(两种口径):
  - direct口径: 多步头直接预测(训练目标, k=1..horizon), 验证集滑窗;
  - rollout口径: rollout_lost 一步预测+自回归递推(部署口径, BeliefManager同款),
    在保留回合(末10%回合, 不参与滑窗方式的训练/验证切分, 见下注)的失观片段上做
    k=1..K 步递推评估, K=min(--rollout-k, 数据中失观段长度).

新增(2026-07-27, M3刷新代码准备, 硬约束6):
  --temp-scale            训练后在验证集(direct口径聚合)拟合单参数温度T(Sigma->T*Sigma)
                          使PICP@95逼近0.95; T以'temperature'键存入checkpoint并报告前后对比.
                          注意: 带'temperature'键的checkpoint被train_belief.load_predictor
                          消费前需先 sd.pop('temperature')(该文件本次按约束未改).
  --scheduled-sampling p  训练时以概率p(按样本)把失观段输入行替换为模型自身递推的
                          伪观测行[mu,0,0,-1](与rollout_lost/BeliefManager部署约定一致),
                          修"训练用真值/部署用伪观测"分布失配(EXPERIMENT_LOG备注③).
  --init ckpt             从已有checkpoint热启动(scheduled sampling式续训入口).
  --rollout-k K           rollout口径最大递推步长(默认50; 部署上限lost_cap_steps=100).
  --fig-out path          per-horizon曲线图(ADE与PICP随k)输出路径, 传空串禁用.
不带新参数时训练流程与旧版一致(切分/迭代随机数流不变; 仅评估报表与图为增量输出).

注: 训练/验证仍沿用旧版"滑窗级"随机切分(保持向后兼容), rollout口径改用末10%回合
以保证整回合评估可行——该保留回合的滑窗可能进过训练集, 存在轻度乐观偏置,
正式刷新时如需严格隔离可改回合级切分(记录在EXPERIMENT_LOG遗留问题).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from belief.gru_predictor import GRUGaussianPredictor  # noqa: E402
from belief.predictor_io import (  # noqa: E402
    load_predictor_checkpoint,
    save_predictor_checkpoint,
)
from belief.transformer_predictor import (  # noqa: E402
    TransformerGaussianPredictor,
)

ARENA = 100.0
VEL_NORM = 8.0


def build_samples(
    npz,
    seq_len=20,
    horizon=5,
    arena=100.0,
    return_episode_ids=False,
    input_mode="legacy_truth",
):
    """Build predictor windows with an explicit lost-observation input policy.

    ``legacy_truth`` reproduces the historical code, including true target
    position in lost rows. ``causal_masked`` removes that leakage: visible rows
    contain measurements, while lost rows contain the last observed position,
    zero velocity, and flag ``-1``. The latter is mandatory for thesis runs.
    """
    if input_mode not in {"legacy_truth", "causal_masked"}:
        raise ValueError(f"unknown input_mode: {input_mode!r}")
    X, Y, episode_ids = [], [], []
    for episode_id, (traj, vel, vis) in enumerate(
        zip(npz["traj"], npz["vel"], npz["vis"])
    ):
        T = len(traj)
        if T < seq_len + horizon + 1:
            continue
        if input_mode == "legacy_truth":
            f = np.concatenate([traj / arena, vel / 8.0, vis[:, None]], -1)
        else:
            f = np.empty((T, 5), dtype=float)
            last_observed = np.asarray(traj[0], dtype=float).copy()
            for t in range(T):
                if vis[t] >= 0.5:
                    last_observed = np.asarray(traj[t], dtype=float).copy()
                    f[t] = [
                        *(np.asarray(traj[t]) / arena),
                        *(np.asarray(vel[t]) / 8.0),
                        1.0,
                    ]
                else:
                    f[t] = [
                        *(last_observed / arena),
                        0.0,
                        0.0,
                        -1.0,
                    ]
        for s in range(0, T - seq_len - horizon, 2):
            X.append(f[s: s + seq_len])
            Y.append(
                np.asarray(
                    traj[s + seq_len: s + seq_len + horizon],
                    dtype=float,
                ) / arena
            )
            episode_ids.append(episode_id)
    result = (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
    )
    if return_episode_ids:
        return (*result, torch.tensor(episode_ids, dtype=torch.long))
    return result


# ---------------- scheduled sampling (分布失配修复) ----------------
def scheduled_sampling_batch(model, xb, p):
    """以概率p(按样本)把该样本窗口内的失观行(vis flag==0)替换为模型自身递推的
    伪观测行 [mu, 0, 0, -1](与 rollout_lost / BeliefManager 部署约定一致).

    - 逐时间位递推: 位置t的伪观测由(已含更早伪观测的)前缀窗口一步预测生成,
      镜像部署期失观段内的自回归;
    - 生成过程 no_grad(标准scheduled sampling, 不对生成输入回传梯度);
    - t=0 的失观行无前缀可推, 保持原样(部署期窗口热身后不存在该情形).
    """
    # Historical datasets use 0 for invisible; deployment-compatible causal
    # masking uses -1. Both must be eligible for scheduled replacement.
    lost = xb[:, :, 4] <= 0.0
    if p <= 0 or not lost.any():
        return xb
    sel = (torch.rand(xb.shape[0], device=xb.device) < p) & lost.any(dim=1)
    if not sel.any():
        return xb
    xm = xb.clone()
    with torch.no_grad():
        for t in range(1, xb.shape[1]):
            rows = sel & lost[:, t]
            if not rows.any():
                continue
            mu, _, _ = model(xm[rows, :t])
            xm[rows, t, 0:2] = mu[:, 0]      # 一步预测均值作伪观测
            xm[rows, t, 2:4] = 0.0
            xm[rows, t, 4] = -1.0            # 部署期失观标志
    return xm


# ---------------- 评估工具 ----------------
def _metrics_per_k(mu, var, y, temperature=None):
    """mu/var/y: (N, K, 2) -> 每档k的 (n, ADE[m], NLL, PICP@95). NLL同nll_loss口径(略常数)."""
    if temperature is not None:
        var = var * temperature
    err = y - mu
    ade = (err ** 2).sum(-1).sqrt().mean(0) * ARENA                        # (K,) 米
    nll = (0.5 * ((err ** 2) / var + var.log()).sum(-1)).mean(0)           # (K,)
    picp = ((err.abs() <= 1.96 * var.sqrt()).all(-1).float()).mean(0)      # (K,)
    n = mu.shape[0]
    return [(n, float(ade[k]), float(nll[k]), float(picp[k])) for k in range(mu.shape[1])]


def aggregate_picp(mu, var, y, temperature=None):
    if temperature is not None:
        var = var * temperature
    return float(((y - mu).abs() <= 1.96 * var.sqrt()).all(-1).float().mean())


def fit_temperature(mu, var, y, target=0.95, iters=60):
    """二分拟合单参数温度T: Sigma->T*Sigma 使 PICP@95 逼近 target(PICP对T单调增)."""
    lo, hi = 1e-4, 1e4
    for _ in range(iters):
        mid = (lo * hi) ** 0.5   # 对数空间二分
        if aggregate_picp(mu, var, y, mid) < target:
            lo = mid
        else:
            hi = mid
    return float((lo * hi) ** 0.5)


def rollout_eval(model, npz, seq_len, k_max, holdout_frac=0.1, temperature=None):
    """部署口径: 末holdout_frac回合上镜像 BeliefManager 管线做失观递推评估.

    逐回合模拟部署期窗口机制(train_belief.py::BeliefManager 同款约定):
    - 窗口热身 = 首帧位置行 [traj0/L, 0, 0, 1] 重复 seq_len 次(镜像reset时
      last_seen_pos=t_pos、t_vel=0 的热身, 与首步真值有一步近似);
    - 可见步压入真实观测行 [pos/L, vel/8, 1];
    - 失观段起点从当前窗口 rollout_lost 一次性递推整段, 第k步预测(mus[k-1])
      对齐真值 traj[s0+k-1], 失观步压入伪观测行 [mu, 0, 0, -1];
    - 指标只统计 k<=k_max. 返回 ({k: (n, ADE[m], NLL, PICP, PICP_T)}, 段数, 回合数).
    """
    n_ep = len(npz["traj"])
    ep_ids = range(max(int(n_ep * (1 - holdout_frac)), 0), n_ep)
    per_k = {}
    n_segs = 0
    for e in ep_ids:
        traj, vel, vis = npz["traj"][e], npz["vel"][e], npz["vis"][e]
        T = len(traj)
        warm = torch.tensor([traj[0][0] / ARENA, traj[0][1] / ARENA, 0.0, 0.0, 1.0],
                            dtype=torch.float32).view(1, 1, 5)
        win = warm.repeat(1, seq_len, 1)
        t = 0
        while t < T:
            if vis[t] >= 0.5:
                row = torch.tensor([*(traj[t] / ARENA), *(vel[t] / VEL_NORM), 1.0],
                                   dtype=torch.float32).view(1, 1, 5)
                win = torch.cat([win[:, 1:], row], dim=1)
                t += 1
                continue
            # 失观段: 一次性递推整段(同BeliefManager新失观事件缓存)
            seg_len = 1
            while t + seg_len < T and vis[t + seg_len] < 0.5:
                seg_len += 1
            n_segs += 1
            mus, vars_ = model.rollout_lost(win, steps=seg_len)   # (1, seg_len, 2)
            y = torch.tensor(traj[t: t + seg_len] / ARENA, dtype=torch.float32)
            for k in range(seg_len):
                if k < k_max:
                    mu_k, var_k, y_k = mus[0, k], vars_[0, k], y[k]
                    err = y_k - mu_k
                    ade = float(err.norm()) * ARENA
                    nll = float(0.5 * ((err ** 2) / var_k + var_k.log()).sum())
                    picp = float((err.abs() <= 1.96 * var_k.sqrt()).all())
                    picp_t = (float((err.abs() <= 1.96 * (var_k * temperature).sqrt()).all())
                              if temperature is not None else None)
                    per_k.setdefault(k + 1, []).append((ade, nll, picp, picp_t))
                pseudo = torch.cat([mus[0, k], torch.zeros(2),
                                    torch.tensor([-1.0])]).view(1, 1, 5)
                win = torch.cat([win[:, 1:], pseudo], dim=1)
            t += seg_len
    out = {}
    for k, rows in sorted(per_k.items()):
        a = np.array([r[:3] for r in rows], float)
        pt = (float(np.mean([r[3] for r in rows]))
              if rows and rows[0][3] is not None else None)
        out[k] = (len(rows), float(a[:, 0].mean()), float(a[:, 1].mean()),
                  float(a[:, 2].mean()), pt)
    return out, n_segs, len(list(ep_ids))


def print_horizon_table(title, rows, has_t=False):
    """rows: [(k, n, ade, nll, picp[, picp_t])]"""
    print(f"\n== {title} ==")
    head = f"{'k':>4} {'n':>7} {'ADE[m]':>8} {'NLL':>9} {'PICP@95':>8}"
    if has_t:
        head += f" {'PICP@95(T)':>11}"
    print(head)
    for r in rows:
        line = f"{r[0]:>4} {r[1]:>7} {r[2]:>8.2f} {r[3]:>9.3f} {r[4]:>8.3f}"
        if has_t:
            line += f" {r[5]:>11.3f}" if r[5] is not None else f" {'-':>11}"
        print(line)


def plot_horizon(fig_out, direct_rows, roll_rows, temperature=None):
    """per-horizon曲线图: ADE与PICP随k(风格参照scripts/plot_curves.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dk = [r[0] for r in direct_rows]
    rk = [r[0] for r in roll_rows]
    axes[0].plot(dk, [r[2] for r in direct_rows], "o-", lw=1.4,
                 color="tab:blue", label="direct多步头(验证集)")
    if roll_rows:
        axes[0].plot(rk, [r[2] for r in roll_rows], "s-", lw=1.4, ms=3,
                     color="tab:red", label="rollout失观递推(保留回合)")
    axes[0].set_xlabel("预测步长 k")
    axes[0].set_ylabel("ADE [m]")
    axes[0].set_title("per-horizon 预测误差")
    axes[1].plot(dk, [r[4] for r in direct_rows], "o-", lw=1.4,
                 color="tab:blue", label="direct多步头")
    if roll_rows:
        axes[1].plot(rk, [r[4] for r in roll_rows], "s-", lw=1.4, ms=3,
                     color="tab:red", label="rollout失观递推")
        if temperature is not None and roll_rows[0][5] is not None:
            axes[1].plot(rk, [r[5] for r in roll_rows], "s--", lw=1.2, ms=3,
                         color="tab:orange", label=f"rollout(温度T={temperature:.3f})")
    axes[1].axhline(0.95, ls=":", lw=1, color="black", alpha=0.7)
    axes[1].annotate("目标0.95", xy=(0.99, 0.951), xycoords=("axes fraction", "data"),
                     ha="right", fontsize=9)
    axes[1].set_xlabel("预测步长 k")
    axes[1].set_ylabel("PICP@95")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("per-horizon 校准(区间覆盖率)")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    Path(fig_out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print("figure saved ->", fig_out)


def build_predictor(args):
    if args.model == "gru":
        return GRUGaussianPredictor(
            in_dim=5,
            hidden=args.hidden,
            horizon=args.horizon,
            rollout_eta=args.rollout_eta,
            rollout_q0=args.rollout_q0,
        )
    return TransformerGaussianPredictor(
        in_dim=5,
        d_model=args.hidden,
        nhead=args.transformer_heads,
        num_layers=args.transformer_layers,
        horizon=args.horizon,
        ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        causal=True,
        rollout_eta=args.rollout_eta,
        rollout_q0=args.rollout_q0,
    )


def save_training_checkpoint(model, path, args, temperature=None):
    """Keep historical GRU files readable while making Transformer self-describing."""
    metadata = {
        "data": str(args.data),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "input_mode": args.input_mode,
        "rollout_eta": args.rollout_eta,
        "rollout_q0": args.rollout_q0,
        "objective": args.loss,
    }
    if args.model == "gru" and args.checkpoint_format == "legacy":
        state_dict = dict(model.state_dict())
        if temperature is not None:
            state_dict["temperature"] = float(temperature)
        torch.save(state_dict, path)
    else:
        save_predictor_checkpoint(
            model,
            path,
            temperature=temperature,
            metadata=metadata,
        )


def make_split_indices(episode_ids, n_episodes, mode, seed=None):
    """Return train, validation, and report indices.

    ``window`` exactly preserves the historical 90/10 random-window split,
    including using the same validation windows for the final report.
    ``episode`` reserves disjoint 80/10/10 episode sets; the final 10% also
    matches ``rollout_eval``'s held-out episode convention.
    """
    n = len(episode_ids)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    if mode == "window":
        idx = torch.randperm(n, generator=generator)
        split = int(0.9 * n)
        return idx[:split], idx[split:], idx[split:]

    train_val_end = max(int(0.9 * n_episodes), 1)
    eligible = torch.arange(train_val_end)
    order = eligible[torch.randperm(len(eligible), generator=generator)]
    train_end = max(int((8.0 / 9.0) * len(order)), 1)
    train_episodes = order[:train_end]
    validation_episodes = order[train_end:]
    report_episodes = torch.arange(train_val_end, n_episodes)
    if not len(validation_episodes) or not len(report_episodes):
        raise ValueError(
            "episode split requires enough episodes for non-empty 80/10/10 sets")
    return tuple(
        torch.nonzero(
            torch.isin(episode_ids, ids), as_tuple=False).squeeze(1)
        for ids in (train_episodes, validation_episodes, report_episodes)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/traj.npz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["nll", "mse"], default="nll",
                    help="概率模型用nll；确定性Transformer对照用mse且不报告校准指标")
    ap.add_argument("--out", default="runs/predictor.pt")
    ap.add_argument("--checkpoint-format", choices=["legacy", "v2"],
                    default="legacy",
                    help="正式论文运行使用v2以保存结构与递推参数；legacy保持旧GRU产物兼容")
    ap.add_argument("--model", choices=["gru", "transformer"], default="gru",
                    help="概率预测器结构；默认GRU保持历史训练入口兼容")
    ap.add_argument("--hidden", type=int, default=64,
                    help="GRU hidden或Transformer d_model")
    ap.add_argument("--transformer-heads", type=int, default=4)
    ap.add_argument("--transformer-layers", type=int, default=2)
    ap.add_argument("--transformer-ff-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rollout-eta", type=float, default=0.05,
                    help="失观递推协方差膨胀斜率；正式值只能在验证集选择")
    ap.add_argument("--rollout-q0", type=float, default=0.04,
                    help="失观递推过程方差；0.04为历史默认，正式值只能在验证集选择")
    ap.add_argument("--split-mode", choices=["window", "episode"],
                    default="window",
                    help="正式论文比较必须使用episode，避免同回合滑窗泄漏")
    ap.add_argument("--input-mode",
                    choices=["legacy_truth", "causal_masked"],
                    default="legacy_truth",
                    help="正式论文比较必须使用causal_masked，禁止失观真值进入输入")
    ap.add_argument("--seed", type=int, default=None,
                    help="显式设置训练/切分随机种子；省略时保持历史随机流")
    ap.add_argument("--temp-scale", action="store_true",
                    help="验证集拟合温度T(Sigma->T*Sigma)使PICP@95->0.95, 存checkpoint['temperature']")
    ap.add_argument("--scheduled-sampling", type=float, default=0.0, metavar="P",
                    help="以概率P把失观段输入行替换为模型递推伪观测[mu,0,0,-1](修分布失配)")
    ap.add_argument("--init", default=None,
                    help="从已有checkpoint热启动(scheduled sampling式续训入口)")
    ap.add_argument("--rollout-k", type=int, default=50,
                    help="rollout口径per-horizon评估的最大递推步长")
    ap.add_argument("--fig-out", default="docs/figs/predictor_horizon.png",
                    help="per-horizon曲线图输出路径(空串禁用)")
    args = ap.parse_args()
    if args.loss == "mse" and args.model != "transformer":
        raise SystemExit("--loss mse仅用于确定性Transformer对照")
    if args.loss == "mse" and args.temp_scale:
        raise SystemExit("确定性Transformer不适用--temp-scale")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    npz = np.load(args.data, allow_pickle=True)
    X, Y, episode_ids = build_samples(
        npz,
        args.seq_len,
        args.horizon,
        return_episode_ids=True,
        input_mode=args.input_mode,
    )
    tr_idx, va_idx, report_idx = make_split_indices(
        episode_ids, len(npz["traj"]), args.split_mode, args.seed)
    print(
        f"samples: train={len(tr_idx)} val={len(va_idx)} "
        f"report={len(report_idx)}, split={args.split_mode}, "
        f"input={args.input_mode}, model={args.model}, device={dev}")
    model = build_predictor(args).to(dev)
    if args.init:
        initialized, _, init_meta = load_predictor_checkpoint(
            args.init, map_location=dev)
        if init_meta["model_type"] != args.model:
            raise SystemExit(
                f"--init模型为{init_meta['model_type']}，但--model={args.model}")
        init_objective = init_meta.get("objective")
        if init_objective is not None and init_objective != args.loss:
            raise SystemExit(
                f"--init目标为{init_objective}，但--loss={args.loss}")
        model.load_state_dict(initialized.state_dict())
        print(f"init from {args.init}")
    if args.scheduled_sampling > 0:
        print(f"scheduled sampling: p={args.scheduled_sampling} "
              f"(失观行替换为模型递推伪观测[mu,0,0,-1])")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X, Y = X.to(dev), Y.to(dev)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for k in range(0, len(perm), args.batch):
            b = perm[k: k + args.batch]
            xb = X[b]
            if args.scheduled_sampling > 0:
                xb = scheduled_sampling_batch(model, xb, args.scheduled_sampling)
            mu, var, _ = model(xb)
            loss = (
                model.nll_loss(mu, var, Y[b])
                if args.loss == "nll"
                else (mu - Y[b]).square().mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            mu, var, _ = model(X[va_idx])
            va_loss = float(
                model.nll_loss(mu, var, Y[va_idx])
                if args.loss == "nll"
                else (mu - Y[va_idx]).square().mean()
            )
            ade = float(((mu - Y[va_idx]) ** 2).sum(-1).sqrt().mean()) * 100.0  # 还原到米
            picp = (
                model.picp(mu, var, Y[va_idx])
                if args.loss == "nll" else float("nan")
            )
        print(
              f"[{ep:3d}] train_{args.loss}={tot/len(tr_idx):.4f} "
              f"val_{args.loss}={va_loss:.4f} "
              f"ADE={ade:.2f}m PICP@95={picp:.3f}")
        if va_loss < best:
            best = va_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(model, args.out, args)
    print(f"best val {args.loss.upper()}:", best, "->", args.out)

    # ---------------- 训练后评估(用best checkpoint) ----------------
    best_model, _, _ = load_predictor_checkpoint(
        args.out, map_location=dev)
    model.load_state_dict(best_model.state_dict())
    model.eval()
    with torch.no_grad():
        calibration_mu, calibration_var, _ = model(X[va_idx])
        mu, var, _ = model(X[report_idx])
    yv = Y[report_idx]

    # 温度缩放仅在验证集拟合，再冻结到report/rollout测试集。
    temperature = None
    if args.temp_scale:
        calibration_y = Y[va_idx]
        temperature = fit_temperature(
            calibration_mu, calibration_var, calibration_y, target=0.95)
        picp0 = aggregate_picp(mu, var, yv)
        picp1 = aggregate_picp(mu, var, yv, temperature)
        with torch.no_grad():
            nll1 = float(model.nll_loss(mu, var * temperature, yv))
        print(f"\n== 温度缩放(验证集拟合，report集报告) ==\n"
              f"T={temperature:.4f}  PICP@95: {picp0:.3f} -> {picp1:.3f} (目标0.95)  "
              f"report_NLL(T): {nll1:.4f}")
        save_training_checkpoint(
            model, args.out, args, temperature=temperature)
        print(f"checkpoint已更新(含temperature) -> {args.out}")

    # per-horizon: direct多步头口径(k=1..horizon, 验证集)
    if args.loss == "nll":
        direct = _metrics_per_k(mu.cpu(), var.cpu(), yv.cpu())
    else:
        error_m = (
            (mu.cpu() - yv.cpu()).square().sum(-1).sqrt().mean(0)
            * ARENA
        )
        direct = [
            (len(mu), float(error_m[k]), float("nan"), float("nan"))
            for k in range(mu.shape[1])
        ]
    direct_rows = [(k + 1, *direct[k]) for k in range(len(direct))]
    if temperature is not None:
        direct_t = _metrics_per_k(mu.cpu(), var.cpu(), yv.cpu(), temperature)
        direct_rows = [(k + 1, *direct[k], direct_t[k][3]) for k in range(len(direct))]
    print_horizon_table(
                        f"per-horizon direct多步头({args.split_mode} report集)",
                        direct_rows,
                        has_t=temperature is not None)

    # per-horizon: rollout失观递推口径(部署口径, 保留回合失观片段)
    model_cpu = model.cpu()
    if args.loss == "nll":
        roll, n_segs, n_hold = rollout_eval(
            model_cpu, npz, args.seq_len, args.rollout_k,
            temperature=temperature)
        roll_rows = [(k, *v) for k, v in roll.items()]
    else:
        roll_rows = []
        n_segs, n_hold = 0, 0
        print(
            "\n[info] 确定性Transformer不使用随机方差；"
            "rollout点误差由eval_thesis_predictor_rollout.py统一评估")
    if roll_rows:
        print_horizon_table(
            f"per-horizon rollout失观递推(保留回合末10%={n_hold}回合, 失观段={n_segs}, "
            f"K<={args.rollout_k})", roll_rows, has_t=temperature is not None)
    else:
        print("\n[warn] 保留回合中无失观段, rollout口径跳过")

    if args.fig_out:
        plot_horizon(args.fig_out, direct_rows, roll_rows, temperature)


if __name__ == "__main__":
    main()
