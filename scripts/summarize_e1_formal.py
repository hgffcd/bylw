# -*- coding: utf-8 -*-
"""按 paper_a_e1_manifest.yaml 汇总E1正式评估，不从训练期eval猜测主表结果。"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml


EVAL_RE = re.compile(
    r"\[eval-seed\s+\d+\s+base=\d+\]\s+"
    r"success=(?P<success>\d+\.\d+)\s+"
    r"collision=(?P<collision>\d+\.\d+)\s+"
    r"timeout=(?P<timeout>\d+\.\d+)\s+"
    r"long_lost=(?P<long_lost>\d+\.\d+)\s+"
    r"mean_len=(?P<mean_len>\d+\.\d+)"
)
METRICS = ("success", "collision", "timeout", "long_lost", "mean_len")


def parse_eval(path: Path, expected_seeds: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    rows = [[float(m.group(k)) for k in METRICS] for m in EVAL_RE.finditer(text)]
    if len(rows) != expected_seeds:
        raise ValueError(f"{path}: expected {expected_seeds} eval rows, got {len(rows)}")
    return np.asarray(rows, float)


def selected_iter(project: Path, run_name: str) -> int:
    path = project / "runs" / run_name / "eval_log.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # train_belief仅在严格大于历史best时覆写best_eval；稳定排序保留首次最大值。
    best = max(enumerate(rows), key=lambda p: (float(p[1]["success"]), -p[0]))[1]
    return int(best["iter"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/paper_a_e1_manifest.yaml")
    ap.add_argument("--result-dir", default="paper_a_results")
    ap.add_argument("--runs-out", default="paper_a_results/e1_formal_runs.csv")
    ap.add_argument("--methods-out", default="paper_a_results/e1_formal_methods.csv")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    with (project / args.manifest).open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    result_dir = project / args.result_dir
    expected_seeds = int(manifest["protocol"]["eval_seeds"])
    run_rows = []
    method_rows = []

    for method, spec in manifest["methods"].items():
        method_metrics = []
        for run in spec["runs"]:
            eval_path = result_dir / run["eval"]
            bins_path = result_dir / run["bins"]
            if not eval_path.exists() or not bins_path.exists():
                raise FileNotFoundError(f"missing formal result for {method}/{run['label']}")
            values = parse_eval(eval_path, expected_seeds)
            means, stds = values.mean(0), values.std(0)
            method_metrics.append(means)
            row = {
                "method": method,
                "label": run["label"],
                "train_seed": run["train_seed"],
                "lineage": run["lineage"],
                "run": run["run"],
                "selected_iter": selected_iter(project, run["run"]),
            }
            for j, metric in enumerate(METRICS):
                row[f"{metric}_mean"] = f"{means[j]:.6f}"
                row[f"{metric}_eval_seed_std"] = f"{stds[j]:.6f}"
            row["eval_file"] = run["eval"]
            row["bins_file"] = run["bins"]
            run_rows.append(row)

        a = np.asarray(method_metrics, float)
        method_row = {
            "method": method,
            "n_runs": len(a),
            "independent_training_seeds": bool(spec["independent_training_seeds"]),
            "success_train_seed_min": f"{a[:, 0].min():.6f}",
            "success_train_seed_max": f"{a[:, 0].max():.6f}",
            "run_success_values": "|".join(f"{v:.6f}" for v in a[:, 0]),
            "note": spec.get("note", ""),
        }
        for j, metric in enumerate(METRICS):
            method_row[f"{metric}_train_seed_mean"] = f"{a[:, j].mean():.6f}"
            method_row[f"{metric}_train_seed_std"] = f"{a[:, j].std():.6f}"
        method_rows.append(method_row)

    runs_out = project / args.runs_out
    methods_out = project / args.methods_out
    runs_out.parent.mkdir(parents=True, exist_ok=True)
    with runs_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)
    with methods_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(method_rows[0]))
        w.writeheader()
        w.writerows(method_rows)
    print(f"runs -> {runs_out}")
    print(f"methods -> {methods_out}")


if __name__ == "__main__":
    main()
