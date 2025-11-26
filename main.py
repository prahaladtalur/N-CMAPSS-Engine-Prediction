"""
Main entry point for N-CMAPSS RUL prediction pipeline.

Supports training and comparing multiple model architectures with wandb logging.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.data.load_data import get_datasets
from src.utils.visualize import visualize_dataset
from src.models.train import train_model, compare_models, list_available_models
from src.models.architectures import get_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="N-CMAPSS RUL Prediction Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train default LSTM model
  python main.py

  # Train specific model
  python main.py --model attention_lstm

  # Compare multiple models
  python main.py --compare --models lstm bilstm attention_lstm tcn

  # Compare all available models
  python main.py --compare

  # Quick training with fewer epochs
  python main.py --model tcn --epochs 20
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help=f"Model architecture to train. Available: {list_available_models()}",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models instead of training a single one",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to compare (default: all available)",
    )
    parser.add_argument(
        "--fd",
        type=int,
        default=1,
        help="FD subset to use (1-7, default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=64,
        help="Number of units in recurrent layers (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip data visualization",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip feature normalization",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model architectures and exit",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="n-cmapss-rul-prediction",
        help="Wandb project name",
    )

    return parser.parse_args()


def list_models_and_exit():
    """Print available models with descriptions."""
    print("\nAvailable Model Architectures:")
    print("=" * 60)

    model_info = get_model_info()
    for name, description in model_info.items():
        print(f"  {name:18} - {description}")

    print("\n" + "=" * 60)
    print("Usage: python main.py --model <model_name>")
    print("       python main.py --compare --models model1 model2 ...")
    sys.exit(0)


def main():
    """Run the complete training pipeline."""
    args = parse_args()

    # List models and exit if requested
    if args.list_models:
        list_models_and_exit()

    print("=" * 60)
    print("N-CMAPSS RUL Prediction Pipeline")
    print("=" * 60)

    # Show configuration
    print("\nConfiguration:")
    print(f"  Dataset: FD00{args.fd}")
    print(f"  Model(s): {args.models if args.compare else args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Units: {args.units}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Wandb project: {args.project}")

    # Step 1: Load data
    print("\n" + "=" * 60)
    print(f"[Step 1] Loading N-CMAPSS FD00{args.fd} dataset...")
    print("=" * 60)
    try:
        (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(
            fd=args.fd, data_dir="data/raw"
        )
        val_X, val_y = val_pair if val_pair else (None, None)
        print(f"Loaded {len(dev_X)} training units, {len(test_X)} test units")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Step 2: Visualize dataset (optional)
    if not args.no_visualize:
        print("\n" + "=" * 60)
        print("[Step 2] Visualizing Dataset")
        print("=" * 60)
        visualize_dataset(
            dev_X=dev_X,
            dev_y=dev_y,
            test_X=test_X,
            test_y=test_y,
            unit_idx=0,
            sensor_indices=[0, 1, 2, 3],
            max_timesteps=500,
        )

    # Training configuration
    train_config = {
        "units": args.units,
        "dense_units": args.units // 2,
        "dropout_rate": 0.2,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_sequence_length": None,
        "validation_split": 0.2,
    }

    # Step 3: Train model(s)
    print("\n" + "=" * 60)
    print("[Step 3] Training Model(s)")
    print("=" * 60)

    if args.compare:
        # Compare multiple models
        models_to_compare = args.models if args.models else list_available_models()
        results = compare_models(
            dev_X=dev_X,
            dev_y=dev_y,
            model_names=models_to_compare,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=train_config,
            project_name=f"{args.project}-comparison",
        )

        print("\n" + "=" * 60)
        print("Comparison Complete!")
        print("=" * 60)
        print("\nResults Summary:")
        for model_name, result in results.items():
            if "metrics" in result:
                rmse = result["metrics"].get("rmse", float("nan"))
                mae = result["metrics"].get("mae", float("nan"))
                print(f"  {model_name:18}: RMSE={rmse:.4f}, MAE={mae:.4f}")
            elif "error" in result:
                print(f"  {model_name:18}: ERROR - {result['error']}")

    else:
        # Train single model
        model, history, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=args.model,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=train_config,
            project_name=args.project,
            run_name=f"{args.model}-fd00{args.fd}",
            normalize=not args.no_normalize,
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Parameters: {model.count_params():,}")
        if metrics:
            print(f"Test RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"Test MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"PHM Score: {metrics.get('phm_score', 'N/A'):.2f}")

    print("\nCheck wandb dashboard for detailed metrics and visualizations")
    print("Results saved to: results/")


if __name__ == "__main__":
    main()
