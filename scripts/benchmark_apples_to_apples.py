#!/usr/bin/env python
"""
Strict apples-to-apples benchmark for top local N-CMAPSS candidates.

All models run with:
- the same FD split
- the same epoch budget
- the same max sequence length
- the same loss / optimizer family
- the same fixed denominator for paper-style normalized metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import get_datasets
from train_model import train_model


DEFAULT_MODELS = [
    "cnn_gru",
    "transformer",
    "mstcn",
    "cata_tcn",
    "wavenet",
]


def rank_key(result: Dict[str, Any]) -> Any:
    metrics = result["metrics"]
    return (
        -(metrics.get("accuracy_20") or float("-inf")),
        metrics.get("rmse") or float("inf"),
        -(metrics.get("accuracy_10") or float("-inf")),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "units": args.units,
        "dense_units": args.dense_units,
        "dropout_rate": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_sequence_length": args.max_sequence_length,
        "loss_name": "asymmetric_mse",
        "loss_alpha": 2.0,
        "optimizer_name": "adam",
        "validation_split": 0.2,
        "patience_early_stop": args.patience_early_stop,
        "patience_lr_reduce": args.patience_lr_reduce,
        "monitor_metric": "val_rmse",
        "monitor_mode": "min",
        "dataset": "ncmapss",
        "reader_max_rul": args.reader_max_rul,
        "feature_set": args.feature_set,
        "resolution_seconds": args.resolution_seconds,
        "sota_target_dataset": "cmapss",
        "fixed_metric_max_rul": args.fixed_metric_max_rul,
    }


def run_benchmark(args: argparse.Namespace) -> List[Dict[str, Any]]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print(f"Loading N-CMAPSS FD{args.fd}...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(
        fd=args.fd,
        max_rul=args.reader_max_rul,
        feature_set=args.feature_set,
        resolution_seconds=args.resolution_seconds,
    )
    val_X, val_y = val_pair if val_pair else (None, None)

    base_config = build_config(args)
    results: List[Dict[str, Any]] = []

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)
        started = time.time()
        _, _, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=model_name,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=base_config,
            project_name=args.project_name,
            run_name=f"a2a_{model_name}_fd{args.fd}_ep{args.epochs}_len{args.max_sequence_length}",
            normalize=True,
            visualize=False,
            save_checkpoint=False,
            seed=args.seed,
        )
        elapsed = time.time() - started
        result = {
            "model": model_name,
            "runtime_seconds": elapsed,
            "config": dict(base_config),
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"{model_name}: Acc@20={metrics['accuracy_20']:.2f}% | "
            f"RMSE={metrics['rmse']:.3f} | "
            f"RMSE_fixed={metrics.get('rmse_normalized_fixed', float('nan')):.4f}"
        )

    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=rank_key)

    (output_dir / "results.json").write_text(
        json.dumps(json_safe(ranked), indent=2),
        encoding="utf-8",
    )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model",
                "accuracy_10",
                "accuracy_15",
                "accuracy_20",
                "rmse",
                "mae",
                "r2",
                "phm_score_normalized",
                "rmse_normalized",
                "rmse_normalized_fixed",
                "mae_normalized_fixed",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(ranked, start=1):
            metrics = result["metrics"]
            writer.writerow(
                {
                    "rank": rank,
                    "model": result["model"],
                    "accuracy_10": metrics.get("accuracy_10"),
                    "accuracy_15": metrics.get("accuracy_15"),
                    "accuracy_20": metrics.get("accuracy_20"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                    "phm_score_normalized": metrics.get("phm_score_normalized"),
                    "rmse_normalized": metrics.get("rmse_normalized"),
                    "rmse_normalized_fixed": metrics.get("rmse_normalized_fixed"),
                    "mae_normalized_fixed": metrics.get("mae_normalized_fixed"),
                    "runtime_seconds": result["runtime_seconds"],
                }
            )

    lines = [
        "# Apples-to-Apples Benchmark",
        "",
        "| Rank | Model | Acc@10 | Acc@20 | RMSE | RMSE(norm,fixed) | R2 | Runtime (min) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(ranked, start=1):
        metrics = result["metrics"]
        lines.append(
            f"| {rank} | {result['model']} | "
            f"{metrics['accuracy_10']:.2f}% | {metrics['accuracy_20']:.2f}% | "
            f"{metrics['rmse']:.3f} | {metrics.get('rmse_normalized_fixed', float('nan')):.4f} | "
            f"{metrics['r2']:.4f} | {result['runtime_seconds']/60:.1f} |"
        )

    best = ranked[0]
    lines.extend(
        [
            "",
            f"Best by Accuracy@20: `{best['model']}` at `{best['metrics']['accuracy_20']:.2f}%`.",
            f"Paper-gap proxy using fixed denominator: `RMSE(norm,fixed)={best['metrics'].get('rmse_normalized_fixed', float('nan')):.4f}`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strict apples-to-apples benchmark.")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--reader-max-rul", type=int, default=65)
    parser.add_argument("--feature-set", choices=["all", "physical"], default="all")
    parser.add_argument("--resolution-seconds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-sequence-length", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience-early-stop", type=int, default=3)
    parser.add_argument("--patience-lr-reduce", type=int, default=2)
    parser.add_argument("--fixed-metric-max-rul", type=float, default=65.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", type=str, default="n-cmapss-a2a")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "apples_to_apples",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir / f"fd{args.fd}_ep{args.epochs}_len{args.max_sequence_length}_{timestamp}"
    )
    results = run_benchmark(args)
    save_results(results, output_dir)
    print(f"\nSaved benchmark to {output_dir}")


if __name__ == "__main__":
    main()
