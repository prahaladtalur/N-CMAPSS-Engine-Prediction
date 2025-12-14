#!/usr/bin/env python
"""
Easy CLI for training and comparing RUL prediction models.

Usage examples:
    # Train a single model
    python train_model.py --model lstm

    # Train with custom config
    python train_model.py --model transformer --epochs 100 --units 128

    # Compare multiple models
    python train_model.py --compare --models lstm gru attention_lstm

    # Compare all models
    python train_model.py --compare-all

    # List available models
    python train_model.py --list-models

    # Get model recommendations
    python train_model.py --recommend
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv is optional
from src.data.load_data import get_datasets
from src.models.train import train_model, compare_models
from src.models.architectures import (
    list_available_models,
    get_model_info,
    get_model_recommendations,
)


def print_models_info():
    """Print information about all available models."""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS FOR RUL PREDICTION")
    print("=" * 80)

    info = get_model_info()

    categories = {
        "RNN-based Models": ["lstm", "bilstm", "gru", "bigru", "attention_lstm", "resnet_lstm"],
        "Convolutional Models": ["tcn", "wavenet"],
        "Hybrid Models": ["cnn_lstm", "cnn_gru", "inception_lstm"],
        "Attention-based Models": ["transformer"],
        "Baseline": ["mlp"],
    }

    for category, model_list in categories.items():
        print(f"\n{category}:")
        for model_name in model_list:
            if model_name in info:
                print(f"  • {model_name:20s} - {info[model_name]}")

    print("\n" + "=" * 80)


def print_recommendations():
    """Print model recommendations for different use cases."""
    print("\n" + "=" * 80)
    print("MODEL RECOMMENDATIONS")
    print("=" * 80)

    recommendations = get_model_recommendations()

    use_cases = {
        "quick_baseline": "Quick Baseline (fast experiments)",
        "best_accuracy": "Best Accuracy (maximum performance)",
        "fastest_training": "Fastest Training (quick iterations)",
        "most_interpretable": "Most Interpretable (understand predictions)",
        "long_sequences": "Long Sequences (many timesteps)",
        "limited_data": "Limited Data (small datasets)",
        "complex_patterns": "Complex Patterns (multi-scale features)",
    }

    for key, description in use_cases.items():
        models = recommendations.get(key, [])
        print(f"\n{description}:")
        print(f"  Recommended: {', '.join(models)}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare RUL prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a single model
  python train_model.py --model lstm

  # Train with custom configuration
  python train_model.py --model transformer --epochs 100 --units 128

  # Compare specific models
  python train_model.py --compare --models lstm gru transformer

  # Compare all available models
  python train_model.py --compare-all

  # List all available models
  python train_model.py --list-models

  # Get model recommendations
  python train_model.py --recommend
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list_available_models(),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list_available_models(),
        help="Multiple models for comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models (use with --models)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all available models",
    )

    # Data options
    parser.add_argument(
        "--fd",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="N-CMAPSS sub-dataset index (default: 1)",
    )

    # Training configuration
    parser.add_argument("--units", type=int, default=64, help="Number of units (default: 64)")
    parser.add_argument(
        "--dense-units", type=int, default=32, help="Dense layer units (default: 32)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: None)",
    )

    # Wandb options
    parser.add_argument(
        "--project",
        type=str,
        default="n-cmapss-rul-prediction",
        help="Wandb project name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Wandb run name (default: model name)",
    )

    # Other options
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature normalization",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show model recommendations and exit",
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_models:
        print_models_info()
        return

    if args.recommend:
        print_recommendations()
        return

    # Validate arguments
    if not args.compare and not args.compare_all and not args.model:
        parser.print_help()
        print("\n\nError: Must specify --model, --compare, or --compare-all")
        sys.exit(1)

    # Load data
    print(f"\nLoading N-CMAPSS FD{args.fd} dataset...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=args.fd)
    val_X, val_y = val_pair if val_pair else (None, None)

    # Training configuration
    config = {
        "units": args.units,
        "dense_units": args.dense_units,
        "dropout_rate": args.dropout,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_sequence_length": args.max_seq_length,
    }

    # Compare mode
    if args.compare or args.compare_all:
        if args.compare_all:
            model_names = list_available_models()
            print(f"\nComparing ALL {len(model_names)} models")
        else:
            model_names = args.models
            if not model_names:
                print("Error: --compare requires --models argument")
                sys.exit(1)

        print(f"Models to compare: {', '.join(model_names)}")

        results = compare_models(
            dev_X=dev_X,
            dev_y=dev_y,
            model_names=model_names,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=config,
            project_name=f"{args.project}-comparison",
        )

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: results/comparison/")
        print(f"Check wandb project: {args.project}-comparison")

    # Single model mode
    else:
        model_name = args.model

        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print("=" * 80)

        model, history, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=model_name,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=config,
            project_name=args.project,
            run_name=args.run_name,
            normalize=not args.no_normalize,
            visualize=not args.no_visualize,
        )

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE: {model_name}")
        print("=" * 80)

        if metrics:
            from src.utils.metrics import format_metrics

            print("\nTest Set Metrics:")
            print(format_metrics(metrics))

        run_name = args.run_name or f"{model_name}-run"
        print(f"\nResults saved to: results/{run_name}/")
        print(f"Check wandb project: {args.project}")


if __name__ == "__main__":
    main()
