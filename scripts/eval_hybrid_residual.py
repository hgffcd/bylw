# -*- coding: utf-8 -*-
"""在独立E5数据上比较IMM、GRU与G1门后hybrid，并做回合集群bootstrap。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from belief.hybrid import hybrid_feature, load_hybrid_checkpoint  # noqa: E402
from belief.imm_filter import IMMFilter  # noqa: E402
from scripts.eval_belief_sources import (  # noqa: E402
    append_metric, eval_gru, rows_from_acc,
)


@torch.no_grad()
def eval_imm_and_hybrid(npz, episode_ids, model, metadata, seq_len: int,
                        max_k: int, arena: float, dt: float,
                        filter_kwargs: dict, acc: dict,
                        details: list) -> int:
    temperature = float(metadata.get("residual_temperature", 1.0))
    segments = 0
    for episode in episode_ids:
        traj = np.asarray(npz["traj"][episode], dtype=float)
        vel = np.asarray(npz["vel"][episode], dtype=float)
        vis = np.asarray(npz["vis"][episode], dtype=float)
        seen = np.flatnonzero(vis >= 0.5)
        if len(seen) == 0:
            continue
        t0 = int(seen[0])
        filt = IMMFilter(**filter_kwargs)
        filt.reset(traj[t0], vel[t0])
        mu, cov = filt.estimate()
        first = hybrid_feature(traj[t0], vel[t0], True, mu, cov,
                               filt.mu_m, 0, arena)
        window = torch.as_tensor(first, dtype=torch.float32).view(
            1, 1, -1).repeat(1, seq_len, 1)
        lost_k, segment = 0, 0
        for t in range(t0 + 1, len(traj)):
            visible = bool(vis[t] >= 0.5)
            if visible:
                mu, cov, probs, _ = filt.update(traj[t], dt)
                lost_k = 0
            else:
                mu, cov, probs = filt.predict(dt)
                if lost_k == 0:
                    segments += 1
                    segment += 1
                lost_k += 1
            feature = hybrid_feature(traj[t], vel[t], visible, mu, cov,
                                     probs, lost_k, arena)
            row = torch.as_tensor(feature, dtype=torch.float32).view(1, 1, -1)
            window = torch.cat([window[:, 1:], row], dim=1)
            if visible or lost_k > max_k:
                continue
            append_metric(acc, "imm", lost_k, traj[t] - mu[:2],
                          cov[:2, :2], details, episode, segment)
            delta, residual_var = model(window)
            hybrid_mu = mu[:2] + delta[0].cpu().numpy() * arena
            hybrid_cov = cov[:2, :2] + np.diag(
                residual_var[0].cpu().numpy() * temperature * (arena ** 2))
            append_metric(acc, "hybrid", lost_k, traj[t] - hybrid_mu,
                          hybrid_cov, details, episode, segment)
    return segments


def bootstrap_episode_differences(details: list, comparisons: list[tuple[str, str]],
                                  replicates: int, seed: int) -> list[dict]:
    by_source: dict[str, dict[int, list[float]]] = {}
    for source, episode, _segment, _k, ade, _nll, _picp in details:
        by_source.setdefault(source, {}).setdefault(int(episode), []).append(float(ade))
    rng = np.random.default_rng(seed)
    rows = []
    for left, right in comparisons:
        episodes = sorted(set(by_source.get(left, {})) &
                          set(by_source.get(right, {})))
        paired = np.asarray([
            np.mean(by_source[left][episode]) -
            np.mean(by_source[right][episode])
            for episode in episodes
        ], dtype=float)
        if not len(paired):
            raise RuntimeError(f"没有{left}-{right}配对回合")
        draws = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sample = rng.integers(0, len(paired), len(paired))
            draws[index] = paired[sample].mean()
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "contrast": f"{left}_minus_{right}",
            "n_episodes": len(paired),
            "mean_ade_difference_m": float(paired.mean()),
            "ci95_low_m": float(low),
            "ci95_high_m": float(high),
            "bootstrap_replicates": replicates,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="data/traj_c1_e5_mix_s1_seed20260729.npz")
    parser.add_argument("--hybrid-ckpt", default="runs/hybrid_residual_r1.pt")
    parser.add_argument("--gru-ckpt", default="runs/predictor_r2.pt")
    parser.add_argument("--filter-config",
                        default="configs/c1_belief_modeswitch_tuned_r4.yaml")
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--arena", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--out",
                        default="paper_a_results/e5_hybrid_quality_independent_mix.csv")
    parser.add_argument("--steps-out",
                        default="paper_a_results/e5_hybrid_quality_independent_mix_steps.csv")
    parser.add_argument("--bootstrap-out",
                        default="paper_a_results/e5_hybrid_bootstrap.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    model, metadata = load_hybrid_checkpoint(args.hybrid_ckpt)
    seq_len = int(metadata.get("seq_len", 20))
    with open(args.filter_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    filter_kwargs = dict(
        (((config.get("train") or {}).get("filter_params") or {}).get("imm") or {}))
    trained_filter = dict(metadata.get("filter_params") or {})
    if trained_filter and trained_filter != filter_kwargs:
        raise ValueError(
            f"hybrid训练IMM参数{trained_filter}与评估参数{filter_kwargs}不一致")
    npz = np.load(args.data, allow_pickle=True)
    episode_ids = list(range(len(npz["traj"])))
    acc, details = {}, []
    hybrid_segments = eval_imm_and_hybrid(
        npz, episode_ids, model, metadata, seq_len, args.max_k,
        args.arena, args.dt, filter_kwargs, acc, details)
    gru_segments = eval_gru(
        npz, args.gru_ckpt, episode_ids, seq_len, args.max_k,
        args.arena, acc, details=details)
    rows = rows_from_acc(acc, ["imm", "gru", "hybrid"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "k", "n", "ade_m", "nll", "picp95"])
        for source, k, n, ade, nll, picp in rows:
            writer.writerow([source, k, n, f"{ade:.6f}",
                             f"{nll:.6f}", f"{picp:.6f}"])
    steps_out = Path(args.steps_out)
    with steps_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "episode", "segment", "k",
                         "ade_m", "nll", "picp95"])
        for row in details:
            writer.writerow([*row[:4], *(f"{value:.6f}" for value in row[4:])])
    bootstrap_rows = bootstrap_episode_differences(
        details, [("hybrid", "imm"), ("hybrid", "gru"), ("imm", "gru")],
        args.bootstrap, args.seed)
    bootstrap_out = Path(args.bootstrap_out)
    with bootstrap_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    print(f"data={args.data} episodes={len(episode_ids)} "
          f"segments hybrid/GRU={hybrid_segments}/{gru_segments}")
    for source, k, n, ade, nll, picp in rows:
        if k == 0:
            print(f"{source:>6}: n={n} ADE={ade:.3f}m "
                  f"NLL={nll:.3f} PICP95={picp:.3f}")
    for row in bootstrap_rows:
        print(f"{row['contrast']}: {row['mean_ade_difference_m']:+.3f}m "
              f"[{row['ci95_low_m']:+.3f},{row['ci95_high_m']:+.3f}]")
    print("summary ->", out)
    print("steps ->", steps_out)
    print("bootstrap ->", bootstrap_out)


if __name__ == "__main__":
    main()
