# -*- coding: utf-8 -*-
"""论文A E5：在同一轨迹数据上比较 CV/KF/IMM/GRU 的失观外推质量。

只评估“目标至少真实可见一次之后”的连续失观段，避免把从未建立过目标状态的
回合混入估计器质量。滤波器参数可从训练配置读取，并与
train_belief.FilterBeliefManager 使用同一参数；
GRU 使用与 BeliefManager 相同的定长窗口、自回归伪观测和温度缩放口径。

输出 CSV:
    source,k,n,ade_m,nll,picp95
其中 k=0 是全部预测步的加权聚合，k>=1 是连续失观段内的步长。

示例:
    python scripts/eval_belief_sources.py \
      --data data/traj_c1_r2.npz --gru-ckpt runs/predictor_r2.pt \
      --max-k 50 --out paper_a_results/e5_belief_quality.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from belief.cv_extrap import CVExtrapolator  # noqa: E402
from belief.imm_filter import IMMFilter  # noqa: E402
from belief.kf_single import KFSingleCV  # noqa: E402
from train_belief import VEL_NORM, load_predictor  # noqa: E402


FILTERS = {
    "cv": CVExtrapolator,
    "kf": KFSingleCV,
    "imm": IMMFilter,
}


def gaussian_metrics(err: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    """二维高斯的欧氏误差、NLL（省略常数）和矩形边缘95%覆盖。"""
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(2)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise FloatingPointError("预测协方差非正定")
    nll = 0.5 * (float(err @ np.linalg.solve(cov, err)) + float(logdet))
    picp = float((np.abs(err) <= 1.96 * np.sqrt(np.diag(cov))).all())
    return float(np.linalg.norm(err)), nll, picp


def append_metric(acc: dict, source: str, k: int,
                  err: np.ndarray, cov: np.ndarray,
                  details: list | None = None, episode: int | None = None,
                  segment: int | None = None) -> None:
    metric = gaussian_metrics(err, cov)
    acc.setdefault((source, k), []).append(metric)
    if details is not None:
        details.append((source, int(episode), int(segment), int(k), *metric))


def eval_filter_source(npz, source: str, ep_ids, dt: float,
                       max_k: int, acc: dict,
                       filter_kwargs: dict | None = None,
                       details: list | None = None) -> int:
    """与 FilterBeliefManager 同口径：可见 update，失观 predict。"""
    n_segments = 0
    for e in ep_ids:
        traj = np.asarray(npz["traj"][e], float)
        vel = np.asarray(npz["vel"][e], float)
        vis = np.asarray(npz["vis"][e], float)
        seen = np.flatnonzero(vis >= 0.5)
        if len(seen) == 0:
            continue
        t0 = int(seen[0])
        filt = FILTERS[source](**dict(filter_kwargs or {}))
        filt.reset(traj[t0], vel[t0])
        lost_k = 0
        segment = 0
        for t in range(t0 + 1, len(traj)):
            if vis[t] >= 0.5:
                filt.update(traj[t], dt)
                lost_k = 0
                continue
            if lost_k == 0:
                n_segments += 1
                segment += 1
            lost_k += 1
            mu, Sigma, _ = filt.predict(dt)
            if lost_k <= max_k:
                append_metric(acc, source, lost_k, traj[t] - mu[:2], Sigma[:2, :2],
                              details, e, segment)
    return n_segments


def _row(pos, vel, flag, arena):
    return torch.tensor([*(pos / arena), *(vel / VEL_NORM), flag],
                        dtype=torch.float32).view(1, 1, 5)


@torch.no_grad()
def eval_gru(npz, ckpt: str, ep_ids, seq_len: int, max_k: int,
             arena: float, acc: dict, details: list | None = None) -> int:
    """与 BeliefManager 同口径：可见推窗口，失观段一次 rollout 后缓存。"""
    model, temperature = load_predictor(ckpt)
    temp = 1.0 if temperature is None else float(temperature)
    n_segments = 0
    for e in ep_ids:
        traj = np.asarray(npz["traj"][e], float)
        vel = np.asarray(npz["vel"][e], float)
        vis = np.asarray(npz["vis"][e], float)
        seen = np.flatnonzero(vis >= 0.5)
        if len(seen) == 0:
            continue
        t = int(seen[0])
        win = _row(traj[t], vel[t], 1.0, arena).repeat(1, seq_len, 1)
        t += 1
        segment = 0
        while t < len(traj):
            if vis[t] >= 0.5:
                win = torch.cat([win[:, 1:], _row(traj[t], vel[t], 1.0, arena)], dim=1)
                t += 1
                continue
            seg_len = 1
            while t + seg_len < len(traj) and vis[t + seg_len] < 0.5:
                seg_len += 1
            n_segments += 1
            segment += 1
            mus, vars_ = model.rollout_lost(win, steps=seg_len)
            for j in range(seg_len):
                if j < max_k:
                    mu_m = mus[0, j].numpy() * arena
                    # GRU方差在归一化位置单位；温度定义为 Sigma -> T*Sigma。
                    var_m2 = vars_[0, j].numpy() * temp * (arena ** 2)
                    append_metric(acc, "gru", j + 1, traj[t + j] - mu_m,
                                  np.diag(var_m2), details, e, segment)
                pseudo = torch.cat([mus[0, j], torch.zeros(2),
                                    torch.tensor([-1.0])]).view(1, 1, 5)
                win = torch.cat([win[:, 1:], pseudo], dim=1)
            t += seg_len
    return n_segments


def rows_from_acc(acc: dict, sources: list[str]) -> list[tuple]:
    rows = []
    for source in sources:
        all_values = []
        ks = sorted(k for s, k in acc if s == source)
        for k in ks:
            a = np.asarray(acc[(source, k)], float)
            all_values.append(a)
            rows.append((source, k, len(a), *a.mean(0)))
        if all_values:
            a = np.concatenate(all_values, axis=0)
            rows.insert(len(rows) - len(ks), (source, 0, len(a), *a.mean(0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/traj_c1_r2.npz")
    ap.add_argument("--gru-ckpt", default="runs/predictor_r2.pt")
    ap.add_argument("--out", default="paper_a_results/e5_belief_quality.csv")
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="使用数据集末尾多少比例的完整回合（默认20%%）")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--arena", type=float, default=100.0)
    ap.add_argument("--filter-config", default=None,
                    help="从 YAML 的 train.filter_params 读取 CV/KF/IMM 参数；"
                         "不传则复现历史默认参数")
    ap.add_argument("--steps-out", default=None,
                    help="可选：输出逐预测步长表(source/episode/segment/k/指标)，"
                         "供回合聚类bootstrap")
    args = ap.parse_args()

    npz = np.load(args.data, allow_pickle=True)
    n_ep = len(npz["traj"])
    start = max(int(n_ep * (1.0 - args.holdout_frac)), 0)
    ep_ids = list(range(start, n_ep))
    filter_params = {}
    if args.filter_config:
        with open(args.filter_config, encoding="utf-8") as f:
            filter_params = ((yaml.safe_load(f) or {}).get("train") or {}).get(
                "filter_params", {}) or {}
    acc = {}
    details = []
    segs = {}
    for source in ("cv", "kf", "imm"):
        segs[source] = eval_filter_source(
            npz, source, ep_ids, args.dt, args.max_k, acc,
            filter_kwargs=filter_params.get(source, {}), details=details)
    segs["gru"] = eval_gru(npz, args.gru_ckpt, ep_ids, args.seq_len,
                            args.max_k, args.arena, acc, details=details)
    sources = ["cv", "kf", "imm", "gru"]
    rows = rows_from_acc(acc, sources)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "k", "n", "ade_m", "nll", "picp95"])
        for source, k, n, ade, nll, picp in rows:
            w.writerow([source, k, n, f"{ade:.6f}", f"{nll:.6f}", f"{picp:.6f}"])
    if args.steps_out:
        steps_out = Path(args.steps_out)
        steps_out.parent.mkdir(parents=True, exist_ok=True)
        with steps_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "episode", "segment", "k",
                        "ade_m", "nll", "picp95"])
            for source, episode, segment, k, ade, nll, picp in details:
                w.writerow([source, episode, segment, k, f"{ade:.6f}",
                            f"{nll:.6f}", f"{picp:.6f}"])

    print(f"data={args.data} holdout episodes={len(ep_ids)}/{n_ep} "
          f"post-visible lost segments={segs}")
    print(f"filter_params={filter_params or 'defaults'}")
    for source, k, n, ade, nll, picp in rows:
        if k == 0:
            print(f"{source:>4}: n={n:6d} ADE={ade:7.3f}m "
                  f"NLL={nll:9.3f} PICP@95={picp:.3f}")
    print("saved ->", out)
    if args.steps_out:
        print("steps ->", args.steps_out)


if __name__ == "__main__":
    main()
