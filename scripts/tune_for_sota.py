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
import time
from pathlib import Path
from typing import Dict, Any, List
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
    fd: int = 1
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
            project_name="sota-tuning",
            run_name=experiment_name,
            visualize=False,  # Speed up experiments
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


def quick_wins_experiments(model_name: str = "mstcn") -> List[Dict]:
    """Test quick configuration changes."""
    
    experiments = []
    
    # Baseline
    baseline_config = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience_early_stop": 10,
        "patience_lr_reduce": 5,
    }
    experiments.append(run_experiment(
        model_name, baseline_config, f"{model_name}_baseline"
    ))
    
    # Longer training
    config_longer = baseline_config.copy()
    config_longer.update({"epochs": 100, "patience_early_stop": 15})
    experiments.append(run_experiment(
        model_name, config_longer, f"{model_name}_longer_training"
    ))
    
    # Smaller batch size (better generalization)
    config_small_batch = baseline_config.copy()
    config_small_batch["batch_size"] = 16
    experiments.append(run_experiment(
        model_name, config_small_batch, f"{model_name}_batch16"
    ))
    
    # Higher learning rate
    config_high_lr = baseline_config.copy()
    config_high_lr["learning_rate"] = 0.002
    experiments.append(run_experiment(
        model_name, config_high_lr, f"{model_name}_lr0.002"
    ))
    
    # Lower learning rate
    config_low_lr = baseline_config.copy()
    config_low_lr["learning_rate"] = 0.0005
    experiments.append(run_experiment(
        model_name, config_low_lr, f"{model_name}_lr0.0005"
    ))
    
    return experiments


def hyperparameter_sweep(model_name: str = "mstcn") -> List[Dict]:
    """Systematic hyperparameter grid search."""
    
    experiments = []
    
    # Grid search key parameters
    learning_rates = [0.0005, 0.001, 0.002]
    batch_sizes = [16, 32, 64]
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            config = {
                "epochs": 75,
                "batch_size": batch_size,
                "learning_rate": lr,
                "patience_early_stop": 12,
                "patience_lr_reduce": 6,
            }
            
            exp_name = f"{model_name}_lr{lr}_bs{batch_size}"
            experiments.append(run_experiment(model_name, config, exp_name))
    
    return experiments


def architecture_comparison() -> List[Dict]:
    """Compare all SOTA architectures with optimized config."""
    
    experiments = []
    
    # Optimized config (to be determined from quick wins)
    optimized_config = {
        "epochs": 100,
        "batch_size": 16,  # Smaller often better
        "learning_rate": 0.001,
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
            model, optimized_config, f"{model}_optimized"
        ))
    
    return experiments


def save_results(experiments: List[Dict], experiment_type: str):
    """Save experiment results to JSON."""
    output_dir = Path("tuning_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{experiment_type}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(experiments, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")
    
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
        choices=["quick_wins", "hyperparams", "architecture"],
        required=True,
        help="Which experiment suite to run"
    )
    parser.add_argument(
        "--model",
        default="mstcn",
        help="Model to tune (for quick_wins and hyperparams)"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "quick_wins":
        print(f"\n🚀 Running quick wins experiments for {args.model}")
        results = quick_wins_experiments(args.model)
        save_results(results, f"quick_wins_{args.model}")
        
    elif args.experiment == "hyperparams":
        print(f"\n🔬 Running hyperparameter sweep for {args.model}")
        results = hyperparameter_sweep(args.model)
        save_results(results, f"hyperparams_{args.model}")
        
    elif args.experiment == "architecture":
        print("\n🏗️  Running architecture comparison with optimized config")
        results = architecture_comparison()
        save_results(results, "architecture_comparison")


if __name__ == "__main__":
    main()
