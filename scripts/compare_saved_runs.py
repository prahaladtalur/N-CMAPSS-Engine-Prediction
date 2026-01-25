#!/usr/bin/env python3
"""
Compare saved W&B runs without retraining and generate a comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.training_viz import plot_model_comparison
import matplotlib.pyplot as plt


DEFAULT_MODELS = ("attention_lstm", "tcn", "cnn_gru")


SUMMARY_KEYS = {
    "rmse": "test_rmse",
    "mae": "test_mae",
    "mse": "test_mse",
    "r2": "test_r2",
    "phm_score": "test_phm_score",
    "phm_score_normalized": "test_phm_score_normalized",
    "accuracy_10": "test_accuracy_10",
    "accuracy_15": "test_accuracy_15",
    "accuracy_20": "test_accuracy_20",
}


METRIC_LINE_RE = re.compile(
    r"^\s*(RMSE|MAE|MSE|R2 Score|PHM Score|PHM Score \(norm\)|Accuracy@10|Accuracy@15|Accuracy@20):\s*([0-9.Ee+-]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load_summary_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    metrics = {}
    for key, summary_key in SUMMARY_KEYS.items():
        if summary_key in data:
            metrics[key] = float(data[summary_key])
    return metrics or None


def parse_output_log(run_dir: Path) -> Optional[Dict[str, float]]:
    log_path = run_dir / "files" / "output.log"
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    metrics = {}
    for line in reversed(lines):
        clean = ANSI_RE.sub("", line).strip()
        match = METRIC_LINE_RE.match(clean)
        if not match:
            continue
        name, value = match.groups()
        value = float(value)
        if name == "RMSE":
            metrics["rmse"] = value
        elif name == "MAE":
            metrics["mae"] = value
        elif name == "MSE":
            metrics["mse"] = value
        elif name == "R2 Score":
            metrics["r2"] = value
        elif name == "PHM Score":
            metrics["phm_score"] = value
        elif name == "PHM Score (norm)":
            metrics["phm_score_normalized"] = value
        elif name == "Accuracy@10":
            metrics["accuracy_10"] = value
        elif name == "Accuracy@15":
            metrics["accuracy_15"] = value
        elif name == "Accuracy@20":
            metrics["accuracy_20"] = value
        if len(metrics) >= 9:
            break
    return metrics or None


def load_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    metrics = load_summary_metrics(run_dir)
    if metrics is not None:
        return metrics
    return parse_output_log(run_dir)


def extract_model_name(run_dir: Path) -> Optional[str]:
    metadata_path = run_dir / "files" / "wandb-metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    args = data.get("args", [])
    for idx, arg in enumerate(args):
        if arg == "--model" and idx + 1 < len(args):
            return str(args[idx + 1])
    return None


def find_latest_runs(models: Sequence[str]) -> Dict[str, Path]:
    run_dirs = sorted(
        Path("wandb").glob("run-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    found: Dict[str, Path] = {}
    for run_dir in run_dirs:
        model_name = extract_model_name(run_dir)
        if model_name is None or model_name not in models:
            continue
        if model_name in found:
            continue
        if load_metrics(run_dir):
            found[model_name] = run_dir
        if len(found) == len(models):
            break
    return found


def discover_models() -> Sequence[str]:
    run_dirs = sorted(
        Path("wandb").glob("run-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    seen = set()
    models = []
    for run_dir in run_dirs:
        model_name = extract_model_name(run_dir)
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        models.append(model_name)
    return models


def is_higher_better(metric: str) -> bool:
    metric = metric.lower()
    if metric in {"r2", "accuracy_10", "accuracy_15", "accuracy_20"}:
        return True
    if metric.startswith("accuracy_"):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare latest saved model runs.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to compare (default: auto-discover).",
    )
    parser.add_argument(
        "--sort-metric",
        type=str,
        default="rmse",
        help="Metric to rank models by (default: rmse).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (higher is better).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show top N models in the leaderboard (0 = all).",
    )
    parser.add_argument(
        "--plot-metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    plt.show = lambda *args, **kwargs: None
    args = parse_args()
    if args.models:
        models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    else:
        models = tuple(discover_models())
    if not models:
        models = DEFAULT_MODELS

    runs = find_latest_runs(models)
    results = {}
    missing = []
    for model_name in models:
        run_dir = runs.get(model_name)
        if not run_dir:
            missing.append(model_name)
            continue
        metrics = load_metrics(run_dir)
        if not metrics:
            missing.append(model_name)
            continue
        results[model_name] = metrics

    if missing:
        print("Missing metrics for:", ", ".join(missing))
    if not results:
        raise SystemExit("No metrics found for any model.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        metrics_to_plot = None
        if args.plot_metrics:
            metrics_to_plot = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
        plot_model_comparison(
            results,
            metrics=metrics_to_plot,
            save_path=str(output_dir / "model_comparison.png"),
        )

    summary_path = output_dir / "model_comparison.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics summary to: {summary_path}")

    sort_metric = args.sort_metric.lower()
    descending = args.descending or is_higher_better(sort_metric)
    missing_value = float("-inf") if descending else float("inf")
    ranked = sorted(
        results.items(),
        key=lambda item: item[1].get(sort_metric, missing_value),
        reverse=descending,
    )
    if args.top and args.top > 0:
        ranked = ranked[: args.top]

    print(f"\nLeaderboard ({sort_metric.upper()}):")
    for model, metrics in ranked:
        metric_value = metrics.get(sort_metric, float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        mae = metrics.get("mae", float("nan"))
        if sort_metric == "rmse":
            print(f"  {model}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        else:
            print(
                f"  {model}: {sort_metric.upper()}={metric_value:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
            )

    best_model, best_metrics = ranked[0]
    print(f"\nBest model by {sort_metric.upper()}: {best_model}")

    metric_keys = list(SUMMARY_KEYS.keys())
    csv_path = output_dir / "model_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *metric_keys])
        writer.writeheader()
        for model, metrics in results.items():
            row = {"model": model}
            row.update({key: metrics.get(key, "") for key in metric_keys})
            writer.writerow(row)
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
