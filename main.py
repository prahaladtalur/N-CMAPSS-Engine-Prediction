"""
Main entry point for N-CMAPSS RUL prediction pipeline.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.data.load_data import get_datasets
from src.utils.visualize import visualize_dataset
from src.models.train import train_lstm


def main():
    """Run the complete data pipeline."""
    print("=" * 60)
    print("N-CMAPSS RUL Prediction Data Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading N-CMAPSS FD001 dataset...")
    try:
        (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=1, data_dir="data/raw")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)

    # Step 2: Visualize dataset (all plots handled inside visualize.py)
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

    # Step 3: Train LSTM model with wandb logging
    print("\n" + "=" * 60)
    print("[Step 3] Training LSTM Model")
    print("=" * 60)

    # Extract validation data if available
    val_X, val_y = val_pair if val_pair else (None, None)

    # Training configuration
    train_config = {
        "lstm_units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "max_sequence_length": None,
        "validation_split": 0.2,
    }

    # Train model
    model, history = train_lstm(
        dev_X=dev_X,
        dev_y=dev_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        config=train_config,
        project_name="n-cmapss-rul-prediction",
        run_name="lstm-baseline-fd001",
    )

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"Model trained with {model.count_params():,} parameters")
    print("Check wandb dashboard for training metrics and visualizations")


if __name__ == "__main__":
    main()
