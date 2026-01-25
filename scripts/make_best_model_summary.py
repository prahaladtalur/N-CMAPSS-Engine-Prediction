#!/usr/bin/env python3
"""
Create a single summary image showing the leaderboard and best model visuals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a best-model summary image.")
    parser.add_argument(
        "--comparison-json",
        type=str,
        default="results/comparison/model_comparison.json",
        help="Path to model comparison JSON.",
    )
    parser.add_argument(
        "--comparison-plot",
        type=str,
        default="results/comparison/model_comparison.png",
        help="Path to model comparison plot image.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory where per-model images live.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rmse",
        help="Metric to select best model (default: rmse).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison/best_model_summary.png",
        help="Output image path.",
    )
    return parser.parse_args()


def is_higher_better(metric: str) -> bool:
    metric = metric.lower()
    if metric in {"r2", "accuracy_10", "accuracy_15", "accuracy_20"}:
        return True
    if metric.startswith("accuracy_"):
        return True
    return False


def load_results(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing comparison JSON: {path}")
    return json.loads(path.read_text())


def select_best_model(results: dict, metric: str) -> str:
    descending = is_higher_better(metric)
    missing_value = float("-inf") if descending else float("inf")
    ranked = sorted(
        results.items(),
        key=lambda item: item[1].get(metric, missing_value),
        reverse=descending,
    )
    if not ranked:
        raise SystemExit("No models found in comparison JSON.")
    return ranked[0][0]


def load_image(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing image: {path}")
    return plt.imread(path)


def main() -> None:
    args = parse_args()
    comparison_json = Path(args.comparison_json)
    comparison_plot = Path(args.comparison_plot)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    results = load_results(comparison_json)
    metric = args.metric.lower()
    best_model = select_best_model(results, metric)

    model_dir = results_dir / f"{best_model}-viz"
    training_path = model_dir / "training_history.png"
    preds_path = model_dir / "predictions.png"
    error_path = model_dir / "error_distribution.png"

    comparison_img = load_image(comparison_plot)
    training_img = load_image(training_path)
    preds_img = load_image(preds_path)
    error_img = load_image(error_path)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Best Model Summary: {best_model} (by {metric.upper()})",
        fontsize=16,
        fontweight="bold",
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.imshow(comparison_img)
    ax0.set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax0.axis("off")

    ax1.imshow(preds_img)
    ax1.set_title(f"{best_model}: Predictions", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(error_img)
    ax2.set_title(f"{best_model}: Error Distribution", fontsize=12, fontweight="bold")
    ax2.axis("off")

    inset = fig.add_axes([0.68, 0.02, 0.3, 0.3])
    inset.imshow(training_img)
    inset.set_title("Training History", fontsize=10, fontweight="bold")
    inset.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary image to: {output_path}")


if __name__ == "__main__":
    main()
