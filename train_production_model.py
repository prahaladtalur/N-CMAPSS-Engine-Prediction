"""
Train and save the production CNN-GRU model with asymmetric loss.

This script trains the best-performing model (CNN-GRU with asymmetric loss)
and saves it for production deployment.
"""

import os
import json
import argparse
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from train_model import train_model, prepare_sequences
from src.data.load_data import get_datasets


def train_production_cnn_gru(
    output_dir: str = "models/production",
    epochs: int = 100,
    max_seq_length: int = 1000,
    run_name: str = "cnn-gru-production",
    fd: int = 1,
) -> str:
    """
    Train CNN-GRU model with optimal hyperparameters and save it.

    Args:
        output_dir: Directory to save the trained model
        epochs: Maximum training epochs
        max_seq_length: Sequence length for truncation
        run_name: Name for this training run
        fd: N-CMAPSS sub-dataset (1-7)

    Returns:
        Path to saved model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Optimal hyperparameters from experiments
    config = {
        "units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": epochs,
        "max_sequence_length": max_seq_length,
        "patience_early_stop": 20,
        "patience_lr_reduce": 8,
    }

    print("=" * 70)
    print("Training Production CNN-GRU Model")
    print("=" * 70)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Load data (train_model will handle preparation and normalization)
    print(f"\nLoading N-CMAPSS FD{fd} dataset...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=fd)
    val_X, val_y = val_pair if val_pair else (None, None)

    # Train the model (it handles all data preparation internally)
    model, history, metrics = train_model(
        dev_X=dev_X,
        dev_y=dev_y,
        model_name="cnn_gru",
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        config=config,
        normalize=True,
        visualize=True,
        run_name=run_name,
    )

    # Save the model
    model_path = os.path.join(output_dir, "cnn_gru_best.keras")
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Save the scaler (re-fit on dev data to match training normalization)
    dev_X_prepared, _ = prepare_sequences(dev_X, dev_y, max_sequence_length=max_seq_length)
    scaler = StandardScaler()
    original_shape = dev_X_prepared.shape
    scaler.fit(dev_X_prepared.reshape(-1, original_shape[-1]))
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")

    print("\n" + "=" * 70)
    print("Production Model Summary")
    print("=" * 70)
    print(f"RMSE:        {metrics['rmse']:.4f}")
    print(f"MAE:         {metrics['mae']:.4f}")
    print(f"PHM Score:   {metrics['phm_score_normalized']:.4f}")
    print(f"R² Score:    {metrics['r2']:.4f}")
    print(f"Acc@10:      {metrics['accuracy_10']:.2f}%")
    print(f"Acc@20:      {metrics['accuracy_20']:.2f}%")
    print("=" * 70)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train production CNN-GRU model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/production",
        help="Directory to save model (default: models/production)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (default: 100)")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1000,
        help="Max sequence length (default: 1000)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="cnn-gru-production",
        help="W&B run name (default: cnn-gru-production)",
    )
    parser.add_argument(
        "--fd",
        type=int,
        default=1,
        help="N-CMAPSS sub-dataset (1-7, default: 1)",
    )

    args = parser.parse_args()

    # Set W&B to offline mode
    os.environ["WANDB_MODE"] = "offline"

    # Train and save model
    model_path = train_production_cnn_gru(
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        run_name=args.run_name,
        fd=args.fd,
    )

    print(f"\n✓ Production model ready at: {model_path}")
