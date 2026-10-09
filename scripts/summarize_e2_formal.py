# -*- coding: utf-8 -*-
"""Summarize frozen E2 ModeSwitch evaluations and join the APF half from E1."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml

try:
    from .summarize_e1_formal import METRICS, parse_eval, selected_iter
except ImportError:
    from summarize_e1_formal import METRICS, parse_eval, selected_iter


SEGMENT_RE = re.compile(r"([VTS])(\d+)")
MODE_NAMES = {"V": "cv", "T": "ct", "S": "stop"}


def read_mode_stats(path: Path, expected_rows: int):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")
    switches = np.asarray([int(r["evader_mode_switches"]) for r in rows], float)
    episode_lengths = np.asarray([int(r["ep_len"]) for r in rows], float)
    counts = {name: 0 for name in MODE_NAMES.values()}
    completed_dwells = {name: [] for name in MODE_NAMES.values()}
    all_modes = 0
    for row in rows:
        segments = [(MODE_NAMES[m], int(n))
                    for m, n in SEGMENT_RE.findall(row["evader_mode_seq"])]
        present = {mode for mode, _ in segments}
        all_modes += int(len(present) == 3)
        for mode, steps in segments:
            counts[mode] += steps
        for mode, steps in segments[:-1]:
            completed_dwells[mode].append(steps)
    total_steps = sum(counts.values())
    return {
        "mode_switches_mean": switches.mean(),
        "mode_switches_std": switches.std(),
        "mode_switches_per_100_steps": (
            100.0 * switches.sum() / episode_lengths.sum()),
        "episodes_all_three_modes": all_modes,
        **{
            f"{mode}_occupancy": counts[mode] / total_steps if total_steps else np.nan
            for mode in ("cv", "ct", "stop")
        },
        **{
            f"{mode}_completed_dwell_steps_mean": (
                float(np.mean(completed_dwells[mode]))
                if completed_dwells[mode] else np.nan
            )
            for mode in ("cv", "ct", "stop")
        },
    }


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/paper_a_e2_manifest.yaml")
    ap.add_argument("--run-map", default="paper_a_results/e2_runs.csv")
    ap.add_argument("--result-dir", default="paper_a_results")
    ap.add_argument("--runs-out", default="paper_a_results/e2_modeswitch_formal_runs.csv")
    ap.add_argument("--methods-out",
                    default="paper_a_results/e2_modeswitch_formal_methods.csv")
    ap.add_argument("--crossover-out",
                    default="paper_a_results/e2_crossover_methods.csv")
    ap.add_argument(
        "--mode-summary-out",
        default="paper_a_results/e2_modeswitch_mode_summary.csv")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    with (project / args.manifest).open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    with (project / args.run_map).open(encoding="utf-8", newline="") as f:
        jobs = list(csv.DictReader(f))
    expected_eval_seeds = int(
        manifest["planned_formal_evaluation"]["eval_seeds"])
    episodes = int(
        manifest["planned_formal_evaluation"]["episodes_per_eval_seed"])
    expected_mode_rows = expected_eval_seeds * episodes
    result_dir = project / args.result_dir

    run_rows = []
    by_source = {}
    for job in jobs:
        source, seed, run = job["source"], int(job["seed"]), job["run_dir"]
        stem = f"e2_{source}_s{seed}"
        eval_file = f"eval_{stem}.txt"
        bins_file = f"bins_{stem}.csv"
        modes_file = f"modes_{stem}.csv"
        for name in (eval_file, bins_file, modes_file):
            if not (result_dir / name).exists():
                raise FileNotFoundError(f"missing formal E2 artifact: {name}")
        values = parse_eval(result_dir / eval_file, expected_eval_seeds)
        means, stds = values.mean(0), values.std(0)
        mode_stats = read_mode_stats(
            result_dir / modes_file, expected_mode_rows)
        row = {
            "evader": "mode_switch",
            "source": source,
            "train_seed": seed,
            "run": run,
            "selected_iter": selected_iter(project, run),
        }
        for j, metric in enumerate(METRICS):
            row[f"{metric}_mean"] = f"{means[j]:.6f}"
            row[f"{metric}_eval_seed_std"] = f"{stds[j]:.6f}"
        for name, value in mode_stats.items():
            row[name] = (
                str(value) if isinstance(value, int) else f"{value:.6f}"
            )
        row.update({
            "eval_file": eval_file,
            "bins_file": bins_file,
            "modes_file": modes_file,
        })
        run_rows.append(row)
        by_source.setdefault(source, []).append(means)

    method_rows = []
    for source in ("kf", "imm", "gru"):
        values = np.asarray(by_source[source], float)
        row = {
            "evader": "mode_switch",
            "source": source,
            "n_runs": len(values),
            "independent_training_seeds": True,
            "run_success_values": "|".join(f"{v:.6f}" for v in values[:, 0]),
        }
        for j, metric in enumerate(METRICS):
            row[f"{metric}_train_seed_mean"] = f"{values[:, j].mean():.6f}"
            row[f"{metric}_train_seed_std"] = f"{values[:, j].std():.6f}"
        method_rows.append(row)

    with (project / "paper_a_results/e1_formal_methods.csv").open(
            encoding="utf-8", newline="") as f:
        e1_rows = {r["method"]: r for r in csv.DictReader(f)}
    crossover_rows = []
    for source in ("kf", "imm", "gru"):
        e1 = e1_rows[source]
        crossover_rows.append({
            "evader": "apf",
            "source": source,
            "n_runs": e1["n_runs"],
            **{
                f"{metric}_train_seed_mean":
                    e1[f"{metric}_train_seed_mean"]
                for metric in METRICS
            },
            **{
                f"{metric}_train_seed_std":
                    e1[f"{metric}_train_seed_std"]
                for metric in METRICS
            },
        })
    for row in method_rows:
        crossover_rows.append({
            key: row[key] for key in crossover_rows[0]
        })

    mode_fields = (
        "mode_switches_mean",
        "mode_switches_per_100_steps",
        "episodes_all_three_modes",
        "cv_occupancy",
        "ct_occupancy",
        "stop_occupancy",
        "cv_completed_dwell_steps_mean",
        "ct_completed_dwell_steps_mean",
        "stop_completed_dwell_steps_mean",
    )
    mode_summary_rows = []
    for source in ("kf", "imm", "gru"):
        rows = [row for row in run_rows if row["source"] == source]
        summary = {
            "source": source,
            "n_training_seeds": len(rows),
            "episodes_per_training_seed": expected_mode_rows,
            "inference_scope": (
                "descriptive_mode_generator_check_across_training_seeds"),
        }
        for field in mode_fields:
            values = np.asarray([float(row[field]) for row in rows], float)
            if field == "episodes_all_three_modes":
                values = values / expected_mode_rows
                output_name = "episodes_all_three_modes_fraction"
            else:
                output_name = field
            summary[f"{output_name}_train_seed_mean"] = (
                f"{values.mean():.6f}")
            summary[f"{output_name}_train_seed_std"] = f"{values.std():.6f}"
        mode_summary_rows.append(summary)

    write_rows(project / args.runs_out, run_rows)
    write_rows(project / args.methods_out, method_rows)
    write_rows(project / args.crossover_out, crossover_rows)
    write_rows(project / args.mode_summary_out, mode_summary_rows)
    print(f"runs -> {project / args.runs_out}")
    print(f"methods -> {project / args.methods_out}")
    print(f"crossover -> {project / args.crossover_out}")
    print(f"mode summary -> {project / args.mode_summary_out}")


if __name__ == "__main__":
    main()
