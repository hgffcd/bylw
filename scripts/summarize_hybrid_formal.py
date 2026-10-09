# -*- coding: utf-8 -*-
"""Summarize formal hybrid runs and apply the frozen hybrid-retention rule."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

try:
    from .summarize_e1_formal import METRICS, parse_eval, selected_iter
    from .summarize_e2_formal import read_mode_stats, write_rows
except ImportError:
    from summarize_e1_formal import METRICS, parse_eval, selected_iter
    from summarize_e2_formal import read_mode_stats, write_rows


SEEDS = (1, 5, 6)


def retention_decision(hybrid, imm, practical_margin=0.05, support_required=2):
    """Apply the preregistered paired-seed rule without episode pseudoreplication."""
    h = np.asarray([hybrid[s] for s in SEEDS], float)
    i = np.asarray([imm[s] for s in SEEDS], float)
    differences = h - i
    support = int((differences >= 0.0).sum())
    mean_difference = float(differences.mean())
    retain = mean_difference >= practical_margin and support >= support_required
    return {
        "retain_hybrid": bool(retain),
        "classification": (
            "hybrid_retained" if retain else "hybrid_not_retained"
        ),
        "primary_method": "hybrid" if retain else "imm",
        "hybrid_mean": float(h.mean()),
        "hybrid_std": float(h.std()),
        "imm_mean": float(i.mean()),
        "imm_std": float(i.std()),
        "hybrid_minus_imm_mean": mean_difference,
        "matched_seed_support": support,
        "paired_differences": differences.tolist(),
    }


def read_imm_runs(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    values = {
        int(row["train_seed"]): float(row["success_mean"])
        for row in rows if row["source"] == "imm"
    }
    if set(values) != set(SEEDS):
        raise ValueError(f"{path}: expected IMM training seeds {SEEDS}")
    return values


def main():
    project = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", default="configs/paper_a_hybrid_manifest.yaml")
    ap.add_argument("--run-map", default="paper_a_results/hybrid_runs.csv")
    ap.add_argument("--result-dir", default="paper_a_results")
    ap.add_argument(
        "--imm-runs",
        default="paper_a_results/e2_modeswitch_formal_runs.csv")
    ap.add_argument(
        "--runs-out", default="paper_a_results/hybrid_formal_runs.csv")
    ap.add_argument(
        "--method-out", default="paper_a_results/hybrid_formal_method.csv")
    ap.add_argument(
        "--evidence-out", default="paper_a_results/hybrid_h1_evidence.csv")
    ap.add_argument(
        "--decision-out", default="paper_a_results/hybrid_h1_decision.md")
    args = ap.parse_args()

    with (project / args.manifest).open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    with (project / args.run_map).open(encoding="utf-8-sig", newline="") as f:
        jobs = list(csv.DictReader(f))
    if len(jobs) != 3 or {int(j["seed"]) for j in jobs} != set(SEEDS):
        raise ValueError("hybrid run map must contain exactly seeds 1, 5, and 6")

    protocol = manifest["closed_loop_test"]["formal_evaluation"]
    expected_eval_seeds = int(protocol["eval_seeds"])
    expected_mode_rows = (
        expected_eval_seeds * int(protocol["episodes_per_eval_seed"])
    )
    result_dir = project / args.result_dir
    run_rows, hybrid_success = [], {}
    for job in sorted(jobs, key=lambda x: int(x["seed"])):
        seed, run = int(job["seed"]), job["run_dir"]
        stem = f"hybrid_s{seed}"
        eval_file = f"eval_{stem}.txt"
        bins_file = f"bins_{stem}.csv"
        modes_file = f"modes_{stem}.csv"
        for name in (eval_file, bins_file, modes_file):
            if not (result_dir / name).exists():
                raise FileNotFoundError(f"missing formal hybrid artifact: {name}")
        values = parse_eval(result_dir / eval_file, expected_eval_seeds)
        means, stds = values.mean(0), values.std(0)
        hybrid_success[seed] = float(means[0])
        row = {
            "evader": "mode_switch",
            "source": "hybrid",
            "train_seed": seed,
            "run": run,
            "selected_iter": selected_iter(project, run),
        }
        for index, metric in enumerate(METRICS):
            row[f"{metric}_mean"] = f"{means[index]:.6f}"
            row[f"{metric}_eval_seed_std"] = f"{stds[index]:.6f}"
        for name, value in read_mode_stats(
                result_dir / modes_file, expected_mode_rows).items():
            row[name] = (
                str(value) if isinstance(value, int) else f"{value:.6f}"
            )
        row.update({
            "eval_file": eval_file,
            "bins_file": bins_file,
            "modes_file": modes_file,
        })
        run_rows.append(row)

    method_row = {
        "evader": "mode_switch",
        "source": "hybrid",
        "n_runs": 3,
        "independent_training_seeds": True,
        "run_success_values": "|".join(
            f"{hybrid_success[s]:.6f}" for s in SEEDS),
    }
    run_metric_values = {
        metric: np.asarray(
            [float(row[f"{metric}_mean"]) for row in run_rows], float)
        for metric in METRICS
    }
    for metric, values in run_metric_values.items():
        method_row[f"{metric}_train_seed_mean"] = f"{values.mean():.6f}"
        method_row[f"{metric}_train_seed_std"] = f"{values.std():.6f}"

    decision_protocol = manifest["closed_loop_test"]["decision"]
    imm_success = read_imm_runs(project / args.imm_runs)
    result = retention_decision(
        hybrid_success,
        imm_success,
        practical_margin=float(decision_protocol["practical_margin"]),
        support_required=int(decision_protocol["matched_seed_support"]),
    )
    evidence = []
    for seed, difference in zip(SEEDS, result["paired_differences"]):
        evidence.append({
            "train_seed": seed,
            "hybrid_success": f"{hybrid_success[seed]:.6f}",
            "imm_success": f"{imm_success[seed]:.6f}",
            "hybrid_minus_imm": f"{difference:.6f}",
            "hybrid_not_worse": bool(difference >= 0.0),
        })

    write_rows(project / args.runs_out, run_rows)
    write_rows(project / args.method_out, [method_row])
    write_rows(project / args.evidence_out, evidence)
    paired = ", ".join(
        f"s{seed}={value:+.3f}"
        for seed, value in zip(SEEDS, result["paired_differences"])
    )
    decision_text = f"""# Paper A post-G1 hybrid decision

Generated after the frozen three-seed, 100 x 3 ModeSwitch evaluation using
`configs/paper_a_hybrid_manifest.yaml`.

## Decision

- Classification: **{result['classification']}**
- Primary method: **{result['primary_method']}**
- Hybrid success: {result['hybrid_mean']:.3f} +/- {result['hybrid_std']:.3f}
- IMM success: {result['imm_mean']:.3f} +/- {result['imm_std']:.3f}
- Hybrid - IMM: {result['hybrid_minus_imm_mean']:+.3f}
- Matched-seed non-worse support: {result['matched_seed_support']}/3
- Matched-seed differences: {paired}

## Frozen rule

Retain hybrid only when mean(Hybrid - IMM) >=
{float(decision_protocol['practical_margin']):.3f} and Hybrid >= IMM for at
least {int(decision_protocol['matched_seed_support'])} of 3 matched training
seeds. Otherwise retain IMM and report hybrid as a neutral or negative
post-G1 ablation.

## Inference boundary

Independent training seeds (n=3), not the 900 evaluation episodes, are the
closed-loop uncertainty units. Offline ADE and calibration results are
supporting evidence and cannot override this preregistered closed-loop rule.
"""
    (project / args.decision_out).write_text(decision_text, encoding="utf-8")
    print(f"hybrid runs -> {project / args.runs_out}")
    print(f"hybrid method -> {project / args.method_out}")
    print(f"hybrid evidence -> {project / args.evidence_out}")
    print(f"hybrid decision -> {project / args.decision_out}")


if __name__ == "__main__":
    main()
