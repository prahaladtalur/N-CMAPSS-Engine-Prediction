#!/usr/bin/env python
"""
Benchmark paper-inspired N-CMAPSS accuracy ideas on a shared training budget.

This script is intentionally pragmatic rather than a full paper reproduction.
It runs a set of model/config variants that map to ideas reported in N-CMAPSS
literature and ranks them by Accuracy@20 on the same FD split.

Usage:
    python scripts/benchmark_paper_ideas.py --fd 1
    python scripts/benchmark_paper_ideas.py --fd 1 --epochs 20
    python scripts/benchmark_paper_ideas.py --only cnn_gru_baseline mstcn_multiscale_conv

Set `WANDB_MODE=offline` only if you explicitly want a local-only run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import get_datasets
from train_model import BEST_ACCURACY_RECIPE, train_model


@dataclass(frozen=True)
class PaperExperiment:
    key: str
    model_name: str
    idea: str
    source_title: str
    source_url: str
    notes: str
    config_overrides: Dict[str, Any]


def build_experiments() -> List[PaperExperiment]:
    """Return the fixed 10-experiment suite."""
    return [
        PaperExperiment(
            key="cnn_gru_baseline",
            model_name="cnn_gru",
            idea="Hybrid convolution + recurrent baseline for local pattern extraction plus temporal memory",
            source_title="An enhanced CNN-LSTM remaining useful life prediction model for aircraft engine with attention mechanism (PeerJ Computer Science, 2022)",
            source_url="https://www.researchgate.net/publication/363110710_An_enhanced_CNN-LSTM_remaining_useful_life_prediction_model_for_aircraft_engine_with_attention_mechanism",
            notes="Family proxy: repo uses CNN-GRU instead of CNN-LSTM-CBAM.",
            config_overrides={},
        ),
        PaperExperiment(
            key="inception_lstm_blocks",
            model_name="inception_lstm",
            idea="Multi-scale inception-style convolutions before temporal modeling",
            source_title="Evaluating Image Classification Deep Convolutional Neural Network Architectures for Remaining Useful Life Estimation of Turbofan Engines (IJPHM, 2022)",
            source_url="https://papers.phmsociety.org/index.php/ijphm/article/view/3284",
            notes="Maps the paper's inception-block idea to the repo's inception-LSTM implementation.",
            config_overrides={},
        ),
        PaperExperiment(
            key="residual_lstm_blocks",
            model_name="resnet_lstm",
            idea="Residual convolutional blocks to improve deeper feature extraction",
            source_title="Evaluating Image Classification Deep Convolutional Neural Network Architectures for Remaining Useful Life Estimation of Turbofan Engines (IJPHM, 2022)",
            source_url="https://papers.phmsociety.org/index.php/ijphm/article/view/3284",
            notes="Maps the paper's ResNet-style block idea to the repo's residual-LSTM implementation.",
            config_overrides={},
        ),
        PaperExperiment(
            key="self_attention_transformer",
            model_name="transformer",
            idea="Self-attention temporal modeling for long-range dependencies",
            source_title="Spatio-temporal degradation modeling and remaining useful life prediction under multiple operating conditions based on attention mechanism and deep learning (RESS, 2022)",
            source_url="https://www.sciencedirect.com/science/article/abs/pii/S0951832022005038",
            notes="Family proxy for transformer-only temporal self-attention.",
            config_overrides={},
        ),
        PaperExperiment(
            key="mdfa_multiscale_fusion_attention",
            model_name="mdfa_paper",
            idea="Multi-scale dilated fusion attention over degradation signals",
            source_title="A spatio-temporal hybrid method with multi-scale BiTCN and modified informer for remaining useful life prediction (Journal of Intelligent Manufacturing, 2025)",
            source_url="https://link.springer.com/article/10.1007/s10845-025-02703-4",
            notes="Family proxy: repo MDFA implementation stands in for recent multi-scale fusion-attention N-CMAPSS methods.",
            config_overrides={},
        ),
        PaperExperiment(
            key="mstcn_multiscale_conv",
            model_name="mstcn",
            idea="Multi-scale temporal convolutions for degradation signals at different horizons",
            source_title="A spatio-temporal hybrid method with multi-scale BiTCN and modified informer for remaining useful life prediction (Journal of Intelligent Manufacturing, 2025)",
            source_url="https://link.springer.com/article/10.1007/s10845-025-02703-4",
            notes="Family proxy: repo MSTCN stands in for the paper's multi-scale temporal-convolution branch.",
            config_overrides={},
        ),
        PaperExperiment(
            key="atcn_attention_tcn",
            model_name="atcn",
            idea="Attention-enhanced temporal convolutions",
            source_title="Spatio-temporal degradation modeling and remaining useful life prediction under multiple operating conditions based on attention mechanism and deep learning (RESS, 2022)",
            source_url="https://www.sciencedirect.com/science/article/abs/pii/S0951832022005038",
            notes="Uses an attention-TCN family member already implemented in the repo.",
            config_overrides={},
        ),
        PaperExperiment(
            key="cata_tcn_channel_temporal_attention",
            model_name="cata_tcn",
            idea="Explicit channel and temporal attention over TCN features",
            source_title="Spatio-temporal degradation modeling and remaining useful life prediction under multiple operating conditions based on attention mechanism and deep learning (RESS, 2022)",
            source_url="https://www.sciencedirect.com/science/article/abs/pii/S0951832022005038",
            notes="Family proxy for separate spatial/channel and temporal attention weighting.",
            config_overrides={},
        ),
        PaperExperiment(
            key="sparse_transformer_bigrcu",
            model_name="sparse_transformer_bigrcu",
            idea="Sparse-attention hybrid encoder for long sequences",
            source_title="LSTM and Transformers based methods for Remaining Useful Life Prediction considering Censored Data (IJPHM, 2025)",
            source_url="https://papers.phmsociety.org/index.php/ijphm/article/view/4260",
            notes="Family proxy: repo uses a sparse transformer hybrid instead of the paper's DAST-style transformer.",
            config_overrides={},
        ),
        PaperExperiment(
            key="cnn_gru_full_accuracy_recipe",
            model_name="cnn_gru",
            idea="Combined training recipe: clipped/scaled targets, AdamW, asymmetric Huber, low-RUL weighting, clipping, calibration",
            source_title="Combination of N-CMAPSS paper ideas: windows/averaging (IJPHM, 2022), self-attention under operating conditions (RESS, 2022), and uncertainty/calibration motivation (arXiv, 2021)",
            source_url="https://papers.phmsociety.org/index.php/ijphm/article/view/3284",
            notes="Repo-native combination recipe rather than a single-paper reproduction.",
            config_overrides=dict(BEST_ACCURACY_RECIPE),
        ),
    ]


def build_base_config(epochs: int) -> Dict[str, Any]:
    """Shared training budget used for all non-recipe runs."""
    return {
        "units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": epochs,
        "max_sequence_length": 1000,
        "loss_name": "asymmetric_mse",
        "loss_alpha": 2.0,
        "validation_split": 0.2,
        "patience_early_stop": max(4, min(8, epochs // 3)),
        "patience_lr_reduce": max(2, min(4, epochs // 5)),
        "monitor_metric": "val_rmse",
        "monitor_mode": "min",
        "dataset": "ncmapss",
        "sota_target_dataset": "cmapss",
    }


def merge_config(
    base_config: Dict[str, Any], overrides: Dict[str, Any], epochs: int
) -> Dict[str, Any]:
    """Merge configs while keeping the CLI epoch budget authoritative."""
    merged = {**base_config, **overrides}
    merged["epochs"] = epochs
    return merged


def rank_key(result: Dict[str, Any]) -> Any:
    """Sort by headline accuracy, then tighter metrics."""
    metrics = result.get("metrics", {})
    return (
        -(metrics.get("accuracy_20") or float("-inf")),
        -(metrics.get("accuracy_10") or float("-inf")),
        metrics.get("rmse") or float("inf"),
        metrics.get("mae") or float("inf"),
    )


def run_suite(
    experiments: Iterable[PaperExperiment],
    *,
    fd: int,
    epochs: int,
    seed: int,
    output_dir: Path,
    project_name: str,
) -> List[Dict[str, Any]]:
    """Run the configured benchmark suite and collect results."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading N-CMAPSS FD{fd} once for all experiments...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=fd)
    val_X, val_y = val_pair if val_pair else (None, None)

    base_config = build_base_config(epochs)
    results: List[Dict[str, Any]] = []

    for index, experiment in enumerate(experiments, start=1):
        print("\n" + "=" * 88)
        print(f"[{index}] {experiment.key} :: {experiment.model_name}")
        print("=" * 88)
        print(f"Idea:   {experiment.idea}")
        print(f"Source: {experiment.source_title}")
        print(f"URL:    {experiment.source_url}")

        config = merge_config(base_config, experiment.config_overrides, epochs)
        run_name = f"{experiment.key}_fd{fd}_ep{epochs}"
        started = time.time()

        try:
            _, _, metrics = train_model(
                dev_X=dev_X,
                dev_y=dev_y,
                model_name=experiment.model_name,
                val_X=val_X,
                val_y=val_y,
                test_X=test_X,
                test_y=test_y,
                config=config,
                project_name=project_name,
                run_name=run_name,
                normalize=True,
                visualize=False,
                save_checkpoint=False,
                seed=seed,
            )
            elapsed = time.time() - started
            results.append(
                {
                    "experiment": asdict(experiment),
                    "status": "ok",
                    "runtime_seconds": elapsed,
                    "metrics": metrics,
                }
            )
            print(
                f"Completed in {elapsed/60:.1f} min | "
                f"Acc@20={metrics.get('accuracy_20', float('nan')):.2f}% | "
                f"RMSE={metrics.get('rmse', float('nan')):.3f}"
            )
        except Exception as exc:  # pragma: no cover - exercised by live runs only
            elapsed = time.time() - started
            results.append(
                {
                    "experiment": asdict(experiment),
                    "status": "error",
                    "runtime_seconds": elapsed,
                    "error": str(exc),
                }
            )
            print(f"Failed after {elapsed/60:.1f} min: {exc}")

    return results


def save_json(results: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")


def save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank",
        "experiment_key",
        "model_name",
        "status",
        "runtime_seconds",
        "accuracy_10",
        "accuracy_15",
        "accuracy_20",
        "rmse",
        "mae",
        "r2",
        "phm_score_normalized",
        "source_title",
        "source_url",
        "notes",
    ]
    ranked = sorted(results, key=rank_key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(ranked, start=1):
            experiment = result["experiment"]
            metrics = result.get("metrics", {})
            writer.writerow(
                {
                    "rank": rank if result["status"] == "ok" else "",
                    "experiment_key": experiment["key"],
                    "model_name": experiment["model_name"],
                    "status": result["status"],
                    "runtime_seconds": f"{result['runtime_seconds']:.2f}",
                    "accuracy_10": metrics.get("accuracy_10"),
                    "accuracy_15": metrics.get("accuracy_15"),
                    "accuracy_20": metrics.get("accuracy_20"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                    "phm_score_normalized": metrics.get("phm_score_normalized"),
                    "source_title": experiment["source_title"],
                    "source_url": experiment["source_url"],
                    "notes": experiment["notes"],
                }
            )


def build_markdown(results: List[Dict[str, Any]], fd: int, epochs: int, seed: int) -> str:
    ranked = sorted(results, key=rank_key)
    lines = [
        "# Paper-Inspired Accuracy Benchmark",
        "",
        f"- Dataset split: N-CMAPSS FD{fd}",
        f"- Epoch budget: {epochs}",
        f"- Seed: {seed}",
        "- Ranking metric: Accuracy@20 descending, then Accuracy@10 descending, then RMSE ascending",
        "",
        "| Rank | Experiment | Model | Acc@10 | Acc@20 | RMSE | MAE | R2 | Runtime (min) | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for rank, result in enumerate(ranked, start=1):
        experiment = result["experiment"]
        metrics = result.get("metrics", {})
        rank_value = str(rank) if result["status"] == "ok" else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    rank_value,
                    experiment["key"],
                    experiment["model_name"],
                    _fmt(metrics.get("accuracy_10"), suffix="%"),
                    _fmt(metrics.get("accuracy_20"), suffix="%"),
                    _fmt(metrics.get("rmse")),
                    _fmt(metrics.get("mae")),
                    _fmt(metrics.get("r2")),
                    f"{result['runtime_seconds'] / 60:.1f}",
                    result["status"],
                ]
            )
            + " |"
        )

    lines.extend(["", "## Sources", ""])
    for result in ranked:
        experiment = result["experiment"]
        lines.append(
            f"- `{experiment['key']}`: [{experiment['source_title']}]({experiment['source_url']})"
        )
        lines.append(f"  Note: {experiment['notes']}")

    winner = next((item for item in ranked if item["status"] == "ok"), None)
    if winner is not None:
        experiment = winner["experiment"]
        metrics = winner["metrics"]
        lines.extend(
            [
                "",
                "## Winner",
                "",
                f"`{experiment['key']}` with `{experiment['model_name']}` delivered the best headline result: "
                f"Accuracy@20={metrics['accuracy_20']:.2f}%, Accuracy@10={metrics['accuracy_10']:.2f}%, "
                f"RMSE={metrics['rmse']:.3f}.",
            ]
        )

    return "\n".join(lines) + "\n"


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2f}{suffix}" if suffix else f"{value:.4f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    """Convert NumPy and nested containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 10 paper-inspired N-CMAPSS accuracy ideas on one split."
    )
    parser.add_argument("--fd", type=int, default=1, help="N-CMAPSS FD split (default: 1)")
    parser.add_argument("--epochs", type=int, default=20, help="Shared epoch budget (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "paper_ideas",
        help="Directory for JSON/CSV/Markdown outputs",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="n-cmapss-paper-ideas",
        help="W&B project name (offline by default)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of experiment keys to run",
    )
    args = parser.parse_args()

    experiments = build_experiments()
    if args.only:
        wanted = set(args.only)
        experiments = [experiment for experiment in experiments if experiment.key in wanted]
        missing = wanted - {experiment.key for experiment in experiments}
        if missing:
            raise SystemExit(f"Unknown experiment keys: {', '.join(sorted(missing))}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"fd{args.fd}_{timestamp}"
    results = run_suite(
        experiments,
        fd=args.fd,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=output_dir,
        project_name=args.project_name,
    )

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    md_path = output_dir / "report.md"
    save_json(results, json_path)
    save_csv(results, csv_path)
    md_path.write_text(build_markdown(results, args.fd, args.epochs, args.seed), encoding="utf-8")

    ranked = [item for item in sorted(results, key=rank_key) if item["status"] == "ok"]
    print("\nSaved:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    if ranked:
        best = ranked[0]
        print(
            "\nBest experiment: "
            f"{best['experiment']['key']} | "
            f"Acc@20={best['metrics']['accuracy_20']:.2f}% | "
            f"RMSE={best['metrics']['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
