"""
Systematic tuning script to close the SOTA gap.

Tests different configurations to beat published SOTA:
- Sequence lengths
- RUL clipping
- Normalization strategies  
- Hyperparameters

Usage:
    python scripts/tune_for_sota.py --experiment quick_wins
    python scripts/tune_for_sota.py --experiment hyperparams
    python scripts/tune_for_sota.py --experiment architecture
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from train_model import train_model
from src.data.load_data import get_datasets

# SOTA target from papers
SOTA_TARGET = 0.032  # RMSE_normalized


def run_experiment(
    model_name: str,
    config: Dict[str, Any],
    experiment_name: str,
    fd: int = 1,
    project_name: str = "sota-tuning",
) -> Dict[str, Any]:
    """Run a single training experiment with given config."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*80}\n")
    
    # Load data
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=fd)
    val_X, val_y = val_pair if val_pair else (None, None)
    
    # Train
    start_time = time.time()
    
    try:
        model, history, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=model_name,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=config,
            project_name=project_name,
            run_name=experiment_name,
            visualize=False,  # Speed up experiments
            save_checkpoint=False,
        )
        
        train_time = time.time() - start_time
        
        # Extract key metrics
        result = {
            "experiment": experiment_name,
            "model": model_name,
            "config": config,
            "rmse_normalized": metrics.get("rmse_normalized", None),
            "mae_normalized": metrics.get("mae_normalized", None),
            "r2": metrics.get("r2", None),
            "accuracy_10": metrics.get("accuracy_10", None),
            "train_time": train_time,
            "success": True,
        }
        
        # Calculate gap from SOTA
        if result["rmse_normalized"]:
            result["sota_gap"] = result["rmse_normalized"] / SOTA_TARGET
            result["improvement_needed"] = result["rmse_normalized"] - SOTA_TARGET
        
        print(f"\n✅ SUCCESS!")
        print(f"   RMSE_norm: {result['rmse_normalized']:.4f}")
        print(f"   SOTA Gap: {result.get('sota_gap', 'N/A'):.2f}x")
        print(f"   Train time: {train_time/60:.1f}m")
        
        return result
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return {
            "experiment": experiment_name,
            "model": model_name,
            "config": config,
            "success": False,
            "error": str(e),
        }


def quick_wins_experiments(
    model_name: str = "mstcn",
    fd: int = 1,
    project_name: str = "sota-tuning",
    epochs: int = 50,
) -> List[Dict]:
    """Test quick configuration changes."""
    
    experiments = []
    
    # Baseline
    baseline_config = {
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": 0.001,
        "loss_name": "asymmetric_mse",
        "loss_alpha": 2.0,
        "max_sequence_length": 1000,
        "patience_early_stop": 10,
        "patience_lr_reduce": 5,
    }
    experiments.append(run_experiment(
        model_name, baseline_config, f"{model_name}_baseline", fd=fd, project_name=project_name
    ))
    
    # Longer training
    config_longer = baseline_config.copy()
    config_longer.update({"epochs": max(epochs * 2, epochs + 10), "patience_early_stop": 15})
    experiments.append(run_experiment(
        model_name, config_longer, f"{model_name}_longer_training", fd=fd, project_name=project_name
    ))
    
    # Smaller batch size (better generalization)
    config_small_batch = baseline_config.copy()
    config_small_batch["batch_size"] = 16
    experiments.append(run_experiment(
        model_name, config_small_batch, f"{model_name}_batch16", fd=fd, project_name=project_name
    ))
    
    # Higher learning rate
    config_high_lr = baseline_config.copy()
    config_high_lr["learning_rate"] = 0.002
    experiments.append(run_experiment(
        model_name, config_high_lr, f"{model_name}_lr0.002", fd=fd, project_name=project_name
    ))
    
    # Lower learning rate
    config_low_lr = baseline_config.copy()
    config_low_lr["learning_rate"] = 0.0005
    experiments.append(run_experiment(
        model_name, config_low_lr, f"{model_name}_lr0.0005", fd=fd, project_name=project_name
    ))

    # Paper-style RUL clipping is common in the literature
    config_rul_clip = baseline_config.copy()
    config_rul_clip["rul_clip_value"] = 125
    experiments.append(run_experiment(
        model_name, config_rul_clip, f"{model_name}_clip125", fd=fd, project_name=project_name
    ))

    # Min-max target scaling can stabilize optimization for long-tailed RUL targets
    config_scaled_targets = baseline_config.copy()
    config_scaled_targets.update({"rul_clip_value": 125, "target_scaling": "minmax"})
    experiments.append(run_experiment(
        model_name,
        config_scaled_targets,
        f"{model_name}_clip125_minmax",
        fd=fd,
        project_name=project_name,
    ))

    # Huber loss is a useful robustness check when large RUL errors dominate training
    config_huber = baseline_config.copy()
    config_huber.update({"rul_clip_value": 125, "target_scaling": "minmax", "loss_name": "huber"})
    experiments.append(run_experiment(
        model_name,
        config_huber,
        f"{model_name}_clip125_minmax_huber",
        fd=fd,
        project_name=project_name,
    ))

    return experiments


def hyperparameter_sweep(
    model_name: str = "mstcn",
    fd: int = 1,
    project_name: str = "sota-tuning",
    epochs: int = 75,
) -> List[Dict]:
    """Systematic hyperparameter grid search."""
    
    experiments = []
    
    # Grid search key parameters
    learning_rates = [0.0005, 0.001, 0.002]
    batch_sizes = [16, 32, 64]
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "rul_clip_value": 125,
                "target_scaling": "minmax",
                "max_sequence_length": 1000,
                "patience_early_stop": 12,
                "patience_lr_reduce": 6,
            }
            
            exp_name = f"{model_name}_lr{lr}_bs{batch_size}"
            experiments.append(
                run_experiment(model_name, config, exp_name, fd=fd, project_name=project_name)
            )
    
    return experiments


def architecture_comparison(
    fd: int = 1,
    project_name: str = "sota-tuning",
    epochs: int = 100,
) -> List[Dict]:
    """Compare all SOTA architectures with optimized config."""
    
    experiments = []
    
    # Optimized config (to be determined from quick wins)
    optimized_config = {
        "epochs": epochs,
        "batch_size": 16,  # Smaller often better
        "learning_rate": 0.001,
        "rul_clip_value": 125,
        "target_scaling": "minmax",
        "loss_name": "huber",
        "gradient_clipnorm": 1.0,
        "max_sequence_length": 1000,
        "patience_early_stop": 15,
        "patience_lr_reduce": 7,
    }
    
    models = [
        "mstcn",
        "mdfa",
        "cnn_lstm_attention",
        "atcn",
        "sparse_transformer_bigrcu",
        "cata_tcn",
        "ttsnet",
    ]
    
    for model in models:
        experiments.append(run_experiment(
            model, optimized_config, f"{model}_optimized", fd=fd, project_name=project_name
        ))
    
    return experiments


def issue25_recipe_experiments(
    model_names: List[str],
    fd: int = 1,
    project_name: str = "sota-tuning",
    epochs: int = 60,
) -> List[Dict]:
    """Run the issue-25 target-preprocessing/training recipe across top models."""
    recipe_configs = [
        (
            "baseline",
            {
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 0.001,
                "loss_name": "asymmetric_mse",
                "loss_alpha": 2.0,
                "max_sequence_length": 1000,
                "patience_early_stop": 10,
                "patience_lr_reduce": 5,
            },
        ),
        (
            "clip125",
            {
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 0.001,
                "loss_name": "asymmetric_mse",
                "loss_alpha": 2.0,
                "rul_clip_value": 125,
                "max_sequence_length": 1000,
                "patience_early_stop": 10,
                "patience_lr_reduce": 5,
            },
        ),
        (
            "clip125_minmax",
            {
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 0.001,
                "loss_name": "asymmetric_mse",
                "loss_alpha": 2.0,
                "rul_clip_value": 125,
                "target_scaling": "minmax",
                "max_sequence_length": 1000,
                "patience_early_stop": 12,
                "patience_lr_reduce": 6,
            },
        ),
        (
            "clip125_minmax_huber",
            {
                "epochs": epochs,
                "batch_size": 16,
                "learning_rate": 0.001,
                "loss_name": "huber",
                "rul_clip_value": 125,
                "target_scaling": "minmax",
                "max_sequence_length": 1000,
                "patience_early_stop": 15,
                "patience_lr_reduce": 7,
            },
        ),
        (
            "clip125_minmax_huber_clipnorm1",
            {
                "epochs": epochs,
                "batch_size": 16,
                "learning_rate": 0.001,
                "loss_name": "huber",
                "rul_clip_value": 125,
                "target_scaling": "minmax",
                "gradient_clipnorm": 1.0,
                "max_sequence_length": 1000,
                "patience_early_stop": 15,
                "patience_lr_reduce": 7,
            },
        ),
    ]

    experiments = []
    for model_name in model_names:
        for suffix, config in recipe_configs:
            experiments.append(
                run_experiment(
                    model_name,
                    config,
                    f"{model_name}_{suffix}",
                    fd=fd,
                    project_name=project_name,
                )
            )
    return experiments


def save_results(experiments: List[Dict], experiment_type: str, fd: int):
    """Save experiment results to JSON."""
    output_dir = Path("tuning_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{experiment_type}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(experiments, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")

    markdown_path = output_dir / f"{experiment_type}_{timestamp}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(f"# Tuning Results: {experiment_type}\n\n")
        f.write(f"- Dataset: FD{fd}\n")
        f.write(f"- Timestamp: {timestamp}\n\n")
        successful = [e for e in experiments if e.get("success", False)]
        if not successful:
            f.write("No successful runs.\n")
        else:
            successful.sort(key=lambda x: x.get("rmse_normalized", float("inf")))
            f.write("| Experiment | Model | RMSE_norm | Gap vs SOTA | R2 |\n")
            f.write("| --- | --- | ---: | ---: | ---: |\n")
            for exp in successful:
                rmse = exp.get("rmse_normalized")
                gap = exp.get("sota_gap")
                r2 = exp.get("r2")
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "N/A"
                gap_str = f"{gap:.2f}x" if isinstance(gap, (int, float)) else "N/A"
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
                f.write(
                    f"| {exp['experiment']} | {exp['model']} | {rmse_str} | {gap_str} | {r2_str} |\n"
                )

    print(f"📝 Markdown summary saved to {markdown_path}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT SUMMARY: {experiment_type}")
    print("="*80)
    
    successful = [e for e in experiments if e.get("success", False)]
    if successful:
        # Sort by RMSE_norm
        successful.sort(key=lambda x: x.get("rmse_normalized", float("inf")))
        
        print(f"\n{'Experiment':<40} {'RMSE_norm':>12} {'SOTA Gap':>10} {'R²':>8}")
        print("-" * 80)
        
        for exp in successful:
            print(
                f"{exp['experiment']:<40} "
                f"{exp.get('rmse_normalized', 'N/A'):>12.4f} "
                f"{exp.get('sota_gap', 'N/A'):>9.2f}x "
                f"{exp.get('r2', 'N/A'):>8.4f}"
            )
        
        # Best result
        best = successful[0]
        print("\n" + "="*80)
        print(f"🏆 BEST: {best['experiment']}")
        print(f"   RMSE_norm: {best['rmse_normalized']:.4f}")
        print(f"   SOTA Gap: {best['sota_gap']:.2f}x")
        print(f"   Improvement needed: {best['improvement_needed']:.4f}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Systematic SOTA tuning")
    parser.add_argument(
        "--experiment",
        choices=["quick_wins", "hyperparams", "architecture", "issue25_recipe"],
        required=True,
        help="Which experiment suite to run"
    )
    parser.add_argument(
        "--model",
        default="mstcn",
        help="Model to tune (for quick_wins and hyperparams)"
    )
    parser.add_argument("--fd", type=int, default=1, help="N-CMAPSS FD split to use")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs per suite")
    parser.add_argument(
        "--project",
        default="sota-tuning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to use for issue25_recipe (default: mstcn transformer wavenet)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run W&B in offline mode for local benchmarking"
    )
    
    args = parser.parse_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        print("\n📴 W&B offline mode enabled")
    
    if args.experiment == "quick_wins":
        print(f"\n🚀 Running quick wins experiments for {args.model}")
        results = quick_wins_experiments(
            args.model,
            fd=args.fd,
            project_name=args.project,
            epochs=args.epochs or 50,
        )
        save_results(results, f"quick_wins_{args.model}", args.fd)
        
    elif args.experiment == "hyperparams":
        print(f"\n🔬 Running hyperparameter sweep for {args.model}")
        results = hyperparameter_sweep(
            args.model,
            fd=args.fd,
            project_name=args.project,
            epochs=args.epochs or 75,
        )
        save_results(results, f"hyperparams_{args.model}", args.fd)
        
    elif args.experiment == "architecture":
        print("\n🏗️  Running architecture comparison with optimized config")
        results = architecture_comparison(
            fd=args.fd,
            project_name=args.project,
            epochs=args.epochs or 100,
        )
        save_results(results, "architecture_comparison", args.fd)
    elif args.experiment == "issue25_recipe":
        model_names = args.models or ["mstcn", "transformer", "wavenet"]
        print(
            f"\n🧪 Running issue-25 recipe on FD{args.fd} for models: {', '.join(model_names)}"
        )
        results = issue25_recipe_experiments(
            model_names,
            fd=args.fd,
            project_name=args.project,
            epochs=args.epochs or 60,
        )
        save_results(results, "issue25_recipe", args.fd)


if __name__ == "__main__":
    main()
