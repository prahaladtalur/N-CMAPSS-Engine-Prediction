"""
Benchmark all SOTA models on N-CMAPSS dataset.

Trains all 7 SOTA architectures and generates comprehensive comparison:
- MDFA (baseline)
- CNN-LSTM-Attention
- CATA-TCN
- TTSNet
- ATCN
- Sparse Transformer + Bi-GRCU
- MSTCN

Usage:
    python scripts/benchmark_sota_models.py --fd 1 --epochs 100
    python scripts/benchmark_sota_models.py --fd 1 --epochs 30 --quick  # Quick test
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. W&B logging disabled.")

# Import training function
import sys

sys.path.append(str(Path(__file__).parent.parent))

from train_model import train_and_evaluate

SOTA_MODELS = [
    "mdfa",
    "cnn_lstm_attention",
    "cata_tcn",
    "ttsnet",
    "atcn",
    "sparse_transformer_bigrcu",
    "mstcn",
]

# SOTA targets from papers
SOTA_TARGETS = {
    "rmse_normalized": 0.032,  # Best from MDFA paper
    "mae_normalized": 0.026,  # Best from literature
    "r2": 0.987,  # High R² target
}


def benchmark_all_models(
    fd: int = 1, epochs: int = 100, visualize: bool = True, wandb_project: str = "n-cmapss-benchmark"
) -> Dict[str, Dict[str, Any]]:
    """
    Train all SOTA models and collect metrics.

    Args:
        fd: N-CMAPSS flight dataset (1-7)
        epochs: Number of training epochs
        visualize: Whether to generate visualizations
        wandb_project: W&B project name for comparison

    Returns:
        Dictionary of {model_name: metrics}
    """
    results = {}
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {len(SOTA_MODELS)} SOTA MODELS ON N-CMAPSS FD{fd}")
    print(f"{'='*80}\n")

    for i, model_name in enumerate(SOTA_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(SOTA_MODELS)}] Training {model_name.upper()}")
        print(f"{'='*80}\n")

        start_time = time.time()

        try:
            # Train model and get metrics
            metrics = train_and_evaluate(
                model_name=model_name,
                fd=fd,
                epochs=epochs,
                visualize=visualize,
                project_name=wandb_project,
            )

            train_time = time.time() - start_time
            metrics["train_time_seconds"] = train_time
            metrics["train_time_formatted"] = format_time(train_time)

            results[model_name] = metrics

            print(f"\n✅ {model_name} completed in {format_time(train_time)}")
            print(f"   RMSE_norm: {metrics.get('rmse_normalized', 'N/A')}")
            print(f"   MAE_norm:  {metrics.get('mae_normalized', 'N/A')}")
            print(f"   R²:        {metrics.get('r2', 'N/A')}")

        except Exception as e:
            print(f"\n❌ {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}

    return results


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_comparison_table(results: Dict[str, Dict[str, Any]], fd: int):
    """Print formatted comparison table."""
    print("\n" + "=" * 130)
    print(f"SOTA MODELS COMPARISON (N-CMAPSS FD{fd})")
    print("=" * 130)

    header = (
        f"{'Model':<30} | {'RMSE_norm':>10} | {'MAE_norm':>10} | "
        f"{'R²':>8} | {'Acc@10':>8} | {'Acc@20':>8} | {'PHM Score':>10} | {'Train Time':>12}"
    )
    print(header)
    print("-" * 130)

    for model_name, metrics in results.items():
        if "error" in metrics:
            row = f"{model_name:<30} | {'ERROR':<80}"
        else:
            row = (
                f"{model_name:<30} | "
                f"{metrics.get('rmse_normalized', 'N/A'):>10.4f} | "
                f"{metrics.get('mae_normalized', 'N/A'):>10.4f} | "
                f"{metrics.get('r2', 'N/A'):>8.4f} | "
                f"{metrics.get('accuracy_10', 'N/A'):>7.2f}% | "
                f"{metrics.get('accuracy_20', 'N/A'):>7.2f}% | "
                f"{metrics.get('phm_score_normalized', 'N/A'):>10.4f} | "
                f"{metrics.get('train_time_formatted', 'N/A'):>12}"
            )
        print(row)

    print("=" * 130)

    # Find best model
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_model = min(
            valid_results.items(), key=lambda x: x[1].get("rmse_normalized", float("inf"))
        )
        best_name, best_metrics = best_model

        print(f"\n🏆 BEST MODEL: {best_name.upper()}")
        print(f"   RMSE_norm: {best_metrics['rmse_normalized']:.4f}")
        print(f"   MAE_norm:  {best_metrics['mae_normalized']:.4f}")
        print(f"   R²:        {best_metrics['r2']:.4f}")
        print(f"   Gap from SOTA target (0.032): {best_metrics['rmse_normalized']/0.032:.2f}x")

        # Check if we beat baseline
        if "mdfa" in valid_results:
            mdfa_rmse = valid_results["mdfa"]["rmse_normalized"]
            improvement = ((mdfa_rmse - best_metrics["rmse_normalized"]) / mdfa_rmse) * 100
            if improvement > 0:
                print(f"\n📈 IMPROVEMENT OVER MDFA: {improvement:.1f}%")
            else:
                print(f"\n📉 MDFA still best (baseline)")
    else:
        print("\n⚠️  No valid results to compare")


def save_results(results: Dict[str, Dict[str, Any]], fd: int, output_dir: str = "benchmark_results"):
    """Save benchmark results to JSON and CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save JSON
    json_file = output_path / f"benchmark_fd{fd}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {json_file}")

    # Save CSV
    csv_data = []
    for model_name, metrics in results.items():
        if "error" not in metrics:
            row = {
                "model": model_name,
                "rmse_normalized": metrics.get("rmse_normalized"),
                "mae_normalized": metrics.get("mae_normalized"),
                "r2": metrics.get("r2"),
                "accuracy_10": metrics.get("accuracy_10"),
                "accuracy_20": metrics.get("accuracy_20"),
                "phm_score_normalized": metrics.get("phm_score_normalized"),
                "train_time_seconds": metrics.get("train_time_seconds"),
            }
            csv_data.append(row)

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = output_path / f"benchmark_fd{fd}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV saved to {csv_file}")


def log_comparison_to_wandb(results: Dict[str, Dict[str, Any]], fd: int, project: str):
    """Log comparison results to W&B dashboard."""
    if not WANDB_AVAILABLE:
        print("\n⚠️  W&B not available, skipping W&B logging")
        return

    wandb.init(project=f"{project}-comparison", name=f"FD{fd}_comparison", reinit=True)

    # Create comparison table
    comparison_data = []
    for model_name, metrics in results.items():
        if "error" not in metrics:
            comparison_data.append(
                [
                    model_name,
                    metrics.get("rmse_normalized"),
                    metrics.get("mae_normalized"),
                    metrics.get("r2"),
                    metrics.get("accuracy_10"),
                    metrics.get("accuracy_20"),
                    metrics.get("train_time_seconds"),
                ]
            )

    comparison_table = wandb.Table(
        columns=[
            "Model",
            "RMSE_norm",
            "MAE_norm",
            "R²",
            "Acc@10",
            "Acc@20",
            "Train Time (s)",
        ],
        data=comparison_data,
    )

    wandb.log({"comparison_table": comparison_table})

    # Log best model info
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_model = min(
            valid_results.items(), key=lambda x: x[1].get("rmse_normalized", float("inf"))
        )
        best_name, best_metrics = best_model

        wandb.summary.update(
            {
                "best_model": best_name,
                "best_rmse_normalized": best_metrics["rmse_normalized"],
                "best_mae_normalized": best_metrics["mae_normalized"],
                "best_r2": best_metrics["r2"],
                "sota_gap": best_metrics["rmse_normalized"] / SOTA_TARGETS["rmse_normalized"],
            }
        )

    wandb.finish()
    print("\n✅ Comparison logged to W&B")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all SOTA models on N-CMAPSS dataset"
    )
    parser.add_argument("--fd", type=int, default=1, choices=range(1, 8), help="Flight dataset (1-7)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization generation"
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument(
        "--wandb-project", type=str, default="n-cmapss-benchmark", help="W&B project name"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test mode (fewer epochs, no visualization)"
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 10)
        args.no_visualize = True
        print("\n⚡ QUICK MODE: Running with reduced epochs and no visualization\n")

    # Run benchmark
    results = benchmark_all_models(
        fd=args.fd,
        epochs=args.epochs,
        visualize=not args.no_visualize,
        wandb_project=args.wandb_project,
    )

    # Print comparison table
    print_comparison_table(results, args.fd)

    # Save results
    save_results(results, args.fd, args.output_dir)

    # Log to W&B
    if WANDB_AVAILABLE:
        log_comparison_to_wandb(results, args.fd, args.wandb_project)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
