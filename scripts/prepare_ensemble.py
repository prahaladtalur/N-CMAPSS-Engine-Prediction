#!/usr/bin/env python3
"""
Prepare ensemble models for production inference.

This script trains and saves the top 3 models (MSTCN, Transformer, WaveNet)
if they don't already exist in models/production/.

Usage:
    # Train all 3 ensemble models
    python scripts/prepare_ensemble.py

    # Force retrain even if models exist
    python scripts/prepare_ensemble.py --force

    # Quick train (10 epochs for testing)
    python scripts/prepare_ensemble.py --quick
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def model_exists(model_name: str, production_dir: str = "models/production") -> bool:
    """Check if a trained model already exists."""
    model_path = Path(production_dir) / f"{model_name}_model.keras"
    metadata_path = Path(production_dir) / f"{model_name}_metadata.json"
    return model_path.exists() and metadata_path.exists()


def train_model(
    model_name: str,
    epochs: int = 30,
    batch_size: int = 64,
    max_seq_length: int = 1000,
    fd: int = 1,
) -> bool:
    """
    Train a single model and save to production directory.

    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}\n")

    cmd = [
        "python",
        "train_model.py",
        "--model",
        model_name,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-seq-length",
        str(max_seq_length),
        "--fd",
        str(fd),
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=False)
        print(f"\n✅ {model_name.upper()} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_name.upper()} training failed: {e}")
        return False


def get_ensemble_models() -> List[Dict]:
    """Get list of ensemble models with their configurations."""
    return [
        {
            "name": "mstcn",
            "display": "MSTCN",
            "weight": 0.5,
            "description": "Multi-Scale TCN + Global Fusion Attention (Best)",
        },
        {
            "name": "transformer",
            "display": "Transformer",
            "weight": 0.3,
            "description": "Multi-head Self-Attention (2nd Best)",
        },
        {
            "name": "wavenet",
            "display": "WaveNet",
            "weight": 0.2,
            "description": "Gated Dilated Convolutions (3rd Best)",
        },
    ]


def save_ensemble_metadata(production_dir: str = "models/production"):
    """Save ensemble configuration metadata."""
    models = get_ensemble_models()

    metadata = {
        "ensemble_name": "top3_ensemble",
        "description": "Ensemble of top 3 models from 20-model benchmark",
        "models": [
            {
                "name": m["name"],
                "weight": m["weight"],
                "display_name": m["display"],
                "model_file": f"{m['name']}_model.keras",
            }
            for m in models
        ],
        "expected_improvement": "10-15% over single best model",
        "usage": "python predict.py --ensemble --fd 1",
        "benchmark_results": {
            "mstcn_rmse": 6.80,
            "transformer_rmse": 6.82,
            "wavenet_rmse": 6.84,
            "ensemble_expected_rmse": "~6.5-6.7 (estimated)",
        },
    }

    os.makedirs(production_dir, exist_ok=True)
    metadata_path = Path(production_dir) / "ensemble_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Ensemble metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ensemble models for production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--force", action="store_true", help="Retrain even if models already exist")

    parser.add_argument(
        "--quick", action="store_true", help="Quick training (10 epochs) for testing"
    )

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")

    parser.add_argument("--fd", type=int, default=1, help="Dataset FD (default: 1)")

    args = parser.parse_args()

    epochs = 10 if args.quick else args.epochs
    production_dir = "models/production"

    print("\n" + "=" * 60)
    print("ENSEMBLE MODEL PREPARATION")
    print("=" * 60)

    models = get_ensemble_models()

    print(f"\nEnsemble Configuration:")
    print(f"  Models: {len(models)}")
    for m in models:
        print(f"    • {m['display']:15s} (weight: {m['weight']:.1f}) - {m['description']}")

    print(f"\nTraining Settings:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: 64")
    print(f"  Sequence Length: 1000")
    print(f"  Dataset: FD{args.fd}")
    print(f"  Production Dir: {production_dir}/")

    # Check existing models
    print(f"\nChecking existing models...")
    existing = []
    missing = []

    for model in models:
        if model_exists(model["name"], production_dir):
            existing.append(model["name"])
            print(f"  ✓ {model['display']:15s} - Found")
        else:
            missing.append(model["name"])
            print(f"  ✗ {model['display']:15s} - Not found")

    if not missing and not args.force:
        print(f"\n✅ All ensemble models already exist!")
        print(f"\nUse --force to retrain, or run predictions with:")
        print(f"  python predict.py --ensemble --fd {args.fd}")
        save_ensemble_metadata(production_dir)
        return

    # Train missing models (or all if --force)
    to_train = [m["name"] for m in models] if args.force else missing

    if not to_train:
        print("\n✅ Nothing to train")
        save_ensemble_metadata(production_dir)
        return

    print(f"\n{'='*60}")
    print(f"Training {len(to_train)} models...")
    print(f"{'='*60}")

    if args.force:
        print("\n⚠️  --force mode: Retraining ALL models")
        response = input("Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return

    print(f"\nEstimated time: ~{len(to_train) * 3} minutes")
    print("(3 minutes per model with early stopping)\n")

    results = {}
    for model_name in to_train:
        success = train_model(
            model_name,
            epochs=epochs,
            batch_size=64,
            max_seq_length=1000,
            fd=args.fd,
        )
        results[model_name] = success

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}\n")

    success_count = sum(results.values())
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {model_name:15s}: {status}")

    print(f"\nTotal: {success_count}/{len(results)} models trained successfully")

    if success_count == len(results):
        print("\n🎉 All models ready for ensemble prediction!")
        print("\nUsage:")
        print(f"  python predict.py --ensemble --fd {args.fd}")
        save_ensemble_metadata(production_dir)
    else:
        print("\n⚠️  Some models failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
