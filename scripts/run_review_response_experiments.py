#!/usr/bin/env python
"""Run controlled experiments requested by paper reviewers.

The goal is to produce evidence for the highest-value review points:

- top-cluster rankings need multiple seeds;
- sequence length needs a single-architecture sweep;
- loss claims need matched loss comparisons;
- virtual-sensor use needs explicit all-feature vs physical-only comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import get_datasets
from train_model import train_model


@dataclass(frozen=True)
class RunSpec:
    key: str
    group: str
    model: str
    seed: int
    fd: int
    feature_set: str = "all"
    reader_max_rul: int = 65
    resolution_seconds: int = 1
    config: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def config_dict(self) -> dict[str, Any]:
        return dict(self.config)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def freeze_config(config: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    frozen_items = []
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            value = tuple(sorted(value.items()))
        elif isinstance(value, list):
            value = tuple(value)
        frozen_items.append((key, value))
    return tuple(frozen_items)


def base_config(
    args: argparse.Namespace, max_sequence_length: int, loss_name: str
) -> dict[str, Any]:
    config = {
        "units": args.units,
        "dense_units": args.dense_units,
        "dropout_rate": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_sequence_length": max_sequence_length,
        "loss_name": loss_name,
        "loss_alpha": args.loss_alpha,
        "loss_delta": args.loss_delta,
        "optimizer_name": "adam",
        "validation_split": 0.2,
        "patience_early_stop": args.patience_early_stop,
        "patience_lr_reduce": args.patience_lr_reduce,
        "monitor_metric": "val_rmse",
        "monitor_mode": "min",
        "dataset": "ncmapss",
        "reader_max_rul": args.reader_max_rul,
        "resolution_seconds": args.resolution_seconds,
        "sota_target_dataset": "ncmapss",
        "compare_to_published_sota": False,
        "fixed_metric_max_rul": args.reader_max_rul,
        "verbose": 0,
    }
    if loss_name == "asymmetric_huber":
        config["target_scaling"] = "minmax"
        config["gradient_clipnorm"] = 1.0
    return config


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: dict[tuple[Any, ...], RunSpec] = {}

    def add(
        *,
        group: str,
        model: str,
        seed: int,
        max_sequence_length: int,
        loss_name: str = "asymmetric_mse",
        feature_set: str = "all",
    ) -> None:
        config = base_config(args, max_sequence_length, loss_name)
        config["feature_set"] = feature_set
        identity = (
            args.fd,
            feature_set,
            args.reader_max_rul,
            args.resolution_seconds,
            model,
            seed,
            freeze_config(config),
        )
        if identity in specs:
            return
        key = (
            f"{group}_{model}_fd{args.fd}_{feature_set}_"
            f"len{max_sequence_length}_{loss_name}_seed{seed}"
        )
        specs[identity] = RunSpec(
            key=key,
            group=group,
            model=model,
            seed=seed,
            fd=args.fd,
            feature_set=feature_set,
            reader_max_rul=args.reader_max_rul,
            resolution_seconds=args.resolution_seconds,
            config=freeze_config(config),
        )

    for seed in args.seeds:
        for model in args.top_models:
            add(group="top_cluster", model=model, seed=seed, max_sequence_length=args.window)

        for length in args.window_sweep:
            add(
                group="window_sweep", model=args.window_model, seed=seed, max_sequence_length=length
            )

        for loss_name in args.losses:
            add(
                group="loss_compare",
                model=args.loss_model,
                seed=seed,
                max_sequence_length=args.window,
                loss_name=loss_name,
            )

        for feature_set in args.feature_sets:
            add(
                group="feature_compare",
                model=args.feature_model,
                seed=seed,
                max_sequence_length=args.window,
                feature_set=feature_set,
            )

    return list(specs.values())


def split_manifest(features: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, Any]:
    labels = np.concatenate([np.asarray(t).reshape(-1) for t in targets])
    first = features[0]
    return {
        "units": len(features),
        "cycles": int(sum(len(f) for f in features)),
        "timesteps": int(first.shape[1]),
        "features": int(first.shape[2]),
        "rul_min": float(labels.min()),
        "rul_max": float(labels.max()),
        "rul_mean": float(labels.mean()),
        "rul_std": float(labels.std()),
    }


def write_outputs(
    results: list[dict[str, Any]], manifests: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(json_safe(results), indent=2))
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifests), indent=2))

    metric_names = [
        "rmse",
        "mae",
        "r2",
        "phm_score_normalized",
        "accuracy_10",
        "accuracy_20",
        "rmse_normalized_fixed",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "key",
                "model",
                "seed",
                "fd",
                "feature_set",
                "reader_max_rul",
                "resolution_seconds",
                "max_sequence_length",
                "loss_name",
                "runtime_seconds",
                *metric_names,
            ],
        )
        writer.writeheader()
        for result in results:
            config = result["config"]
            row = {
                "group": result["group"],
                "key": result["key"],
                "model": result["model"],
                "seed": result["seed"],
                "fd": result["fd"],
                "feature_set": result["feature_set"],
                "reader_max_rul": result["reader_max_rul"],
                "resolution_seconds": result["resolution_seconds"],
                "max_sequence_length": config["max_sequence_length"],
                "loss_name": config["loss_name"],
                "runtime_seconds": result["runtime_seconds"],
            }
            row.update({name: result["metrics"].get(name) for name in metric_names})
            writer.writerow(row)

    first_manifest = next(iter(manifests.values()))
    feature_sets = sorted({manifest["feature_set"] for manifest in manifests.values()})
    max_ruls = sorted({manifest["reader_max_rul"] for manifest in manifests.values()})
    resolutions = sorted({manifest["resolution_seconds"] for manifest in manifests.values()})

    lines = [
        "# Review-Response Controlled Experiments",
        "",
        "These runs address reviewer concerns about seed variance, sequence-length confounding, loss comparisons, and virtual-sensor use.",
        "",
        "## Dataset",
        "",
        f"- N-CMAPSS fd={first_manifest['fd']} via `rul-datasets`.",
        "- Reader split: default `rul-datasets` split, with the last 20% of original training units used for validation.",
        f"- Reader max RUL values: {', '.join(str(value) for value in max_ruls)} cycles.",
        f"- Resolution values: {', '.join(str(value) for value in resolutions)} second(s).",
        f"- Feature sets evaluated: {', '.join(feature_sets)}.",
        "- `all` = 4 operating conditions + 14 physical sensors + 14 virtual sensors; `physical` = no virtual sensors.",
        "",
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["group"], []).append(result)

    # Some settings are intentionally deduplicated across experiment groups.
    # Reuse those completed reference rows in the report so each review question
    # has a complete comparison table without retraining identical runs.
    for result in results:
        config = result["config"]
        is_reference_wavenet = (
            result["model"] == "wavenet"
            and result["feature_set"] == "all"
            and config["max_sequence_length"] == 1000
            and config["loss_name"] == "asymmetric_mse"
        )
        if is_reference_wavenet:
            grouped.setdefault("loss_compare", []).append({**result, "group": "loss_compare"})
            grouped.setdefault("feature_compare", []).append({**result, "group": "feature_compare"})

    for group, items in grouped.items():
        lines.extend(
            [
                f"## {group}",
                "",
                "| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in items:
            config = item["config"]
            setting = (
                item["model"],
                item["feature_set"],
                config["max_sequence_length"],
                config["loss_name"],
            )
            buckets.setdefault(setting, []).append(item)
        for (model, feature_set, length, loss_name), bucket in sorted(buckets.items()):
            rmses = [b["metrics"]["rmse"] for b in bucket]
            acc20 = [b["metrics"]["accuracy_20"] for b in bucket]
            r2s = [b["metrics"]["r2"] for b in bucket]
            rmse_std = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
            setting_text = f"{feature_set}, T={length}, {loss_name}"
            lines.append(
                f"| {model} | {setting_text} | {len(bucket)} | "
                f"{np.mean(rmses):.3f} | {rmse_std:.3f} | "
                f"{np.mean(acc20):.2f}% | {np.mean(r2s):.4f} |"
            )
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_outputs(output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load previously saved results and manifests for a resumed run."""
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"

    results: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifests = json.load(handle)
    return results, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--reader-max-rul", type=int, default=65)
    parser.add_argument("--resolution-seconds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--window-sweep", type=int, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--window-model", default="wavenet")
    parser.add_argument("--loss-model", default="wavenet")
    parser.add_argument("--feature-model", default="wavenet")
    parser.add_argument("--top-models", nargs="+", default=["wavenet", "cnn_gru", "mstcn"])
    parser.add_argument("--feature-sets", nargs="+", default=["all", "physical"])
    parser.add_argument(
        "--losses", nargs="+", default=["mse", "asymmetric_mse", "asymmetric_huber"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--loss-alpha", type=float, default=2.0)
    parser.add_argument("--loss-delta", type=float, default=0.08)
    parser.add_argument("--patience-early-stop", type=int, default=6)
    parser.add_argument("--patience-lr-reduce", type=int, default=3)
    parser.add_argument("--project-name", default="n-cmapss-review-response")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "review_response",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Existing fd*_review_* output directory to resume into.",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    specs = build_specs(args)
    if args.resume_dir is not None:
        output_dir = args.resume_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"fd{args.fd}_review_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, Any] = {}
    results, manifests = load_existing_outputs(output_dir) if args.resume_dir else ([], {})
    completed_keys = {result["key"] for result in results}

    print(f"Prepared {len(specs)} unique runs.")
    if completed_keys:
        print(f"Resuming with {len(completed_keys)} completed runs already saved.")
    for index, spec in enumerate(specs, start=1):
        if spec.key in completed_keys:
            print(f"[{index}/{len(specs)}] skipping completed {spec.key}")
            continue

        dataset_key = f"fd{spec.fd}_{spec.feature_set}_maxrul{spec.reader_max_rul}_res{spec.resolution_seconds}"
        if dataset_key not in datasets:
            print(f"\nLoading dataset variant: {dataset_key}")
            dataset = get_datasets(
                fd=spec.fd,
                max_rul=spec.reader_max_rul,
                feature_set=spec.feature_set,
                resolution_seconds=spec.resolution_seconds,
            )
            datasets[dataset_key] = dataset
            (dev_X, dev_y), val_pair, (test_X, test_y) = dataset
            manifests[dataset_key] = {
                "fd": spec.fd,
                "feature_set": spec.feature_set,
                "reader_max_rul": spec.reader_max_rul,
                "resolution_seconds": spec.resolution_seconds,
                "dev": split_manifest(dev_X, dev_y),
                "val": split_manifest(*val_pair) if val_pair else None,
                "test": split_manifest(test_X, test_y),
            }

        (dev_X, dev_y), val_pair, (test_X, test_y) = datasets[dataset_key]
        val_X, val_y = val_pair if val_pair else (None, None)
        config = spec.config_dict()

        print("\n" + "=" * 88)
        print(f"[{index}/{len(specs)}] {spec.key}")
        print("=" * 88)
        started = time.time()
        _, _, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=spec.model,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=config,
            project_name=args.project_name,
            run_name=spec.key,
            normalize=True,
            visualize=False,
            save_checkpoint=False,
            seed=spec.seed,
        )
        runtime = time.time() - started
        result = {
            "key": spec.key,
            "group": spec.group,
            "model": spec.model,
            "seed": spec.seed,
            "fd": spec.fd,
            "feature_set": spec.feature_set,
            "reader_max_rul": spec.reader_max_rul,
            "resolution_seconds": spec.resolution_seconds,
            "config": config,
            "metrics": metrics,
            "runtime_seconds": runtime,
        }
        results.append(json_safe(result))
        write_outputs(results, manifests, output_dir)
        print(f"Saved interim results to {output_dir}")

    write_outputs(results, manifests, output_dir)
    print(f"\nComplete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
