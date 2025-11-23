"""
Training pipeline for LSTM RUL prediction model.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback

from src.models.lstm_model import build_lstm_model


def prepare_sequences(
    X: List[np.ndarray], y: List[np.ndarray], max_sequence_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences from list of arrays for training.

    Args:
        X: List of feature arrays (num_cycles, timesteps, num_sensors)
        y: List of label arrays (num_cycles,)
        max_sequence_length: Maximum sequence length (None for full sequences)

    Returns:
        Tuple of (X_sequences, y_sequences) as numpy arrays
    """
    X_sequences = []
    y_sequences = []

    for unit_X, unit_y in zip(X, y):
        num_cycles = unit_X.shape[0]
        for cycle_idx in range(num_cycles):
            sequence = unit_X[cycle_idx]
            if max_sequence_length is not None:
                # Take the last max_sequence_length timesteps
                sequence = sequence[-max_sequence_length:]
            X_sequences.append(sequence)
            y_sequences.append(unit_y[cycle_idx])

    return np.array(X_sequences), np.array(y_sequences)


def train_lstm(
    dev_X: List[np.ndarray],
    dev_y: List[np.ndarray],
    val_X: Optional[List[np.ndarray]] = None,
    val_y: Optional[List[np.ndarray]] = None,
    test_X: Optional[List[np.ndarray]] = None,
    test_y: Optional[List[np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    project_name: str = "n-cmapss-rul-prediction",
    run_name: Optional[str] = None,
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Train LSTM model with wandb logging.

    Args:
        dev_X: Development set features
        dev_y: Development set labels
        val_X: Validation set features (optional)
        val_y: Validation set labels (optional)
        test_X: Test set features (optional, for final evaluation)
        test_y: Test set labels (optional, for final evaluation)
        config: Training configuration dictionary
        project_name: Wandb project name
        run_name: Wandb run name

    Returns:
        Tuple of (trained_model, training_history_dict)
    """
    # Default configuration
    default_config = {
        "lstm_units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "max_sequence_length": None,
        "validation_split": 0.2,
    }

    if config is None:
        config = default_config
    else:
        # Merge with defaults
        config = {**default_config, **config}

    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True,
    )

    # Prepare training data
    print("Preparing training sequences...")
    X_train, y_train = prepare_sequences(dev_X, dev_y, config["max_sequence_length"])
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Prepare validation data
    if val_X is not None and val_y is not None:
        X_val, y_val = prepare_sequences(val_X, val_y, config["max_sequence_length"])
        print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        validation_data = (X_val, y_val)
    else:
        validation_data = None
        print("No validation set provided, using validation_split")

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=config["lstm_units"],
        dense_units=config["dense_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )

    print("\nModel Architecture:")
    model.summary()

    # Log model architecture to wandb
    wandb.config.update({"input_shape": input_shape, "model_params": model.count_params()})

    # Callbacks
    callbacks = [
        WandbCallback(
            monitor="val_loss" if validation_data else "loss",
            log_weights=True,
            log_gradients=True,
            save_model=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=validation_data,
        validation_split=config["validation_split"] if validation_data is None else None,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set if provided
    test_metrics = {}
    if test_X is not None and test_y is not None:
        print("\nEvaluating on test set...")
        X_test, y_test = prepare_sequences(test_X, test_y, config["max_sequence_length"])
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_metrics = {
            "test_loss": test_results[0],
            "test_mae": test_results[1],
            "test_mape": test_results[2],
        }
        print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        print(f"Test MAE: {test_metrics['test_mae']:.4f}")
        print(f"Test MAPE: {test_metrics['test_mape']:.4f}")

        # Log test metrics to wandb
        wandb.log(test_metrics)

    # Log final metrics
    final_metrics = {
        "final_train_loss": history.history["loss"][-1],
        "final_train_mae": history.history["mae"][-1],
    }
    if validation_data:
        final_metrics["final_val_loss"] = history.history["val_loss"][-1]
        final_metrics["final_val_mae"] = history.history["val_mae"][-1]

    wandb.log(final_metrics)
    wandb.finish()

    return model, {**history.history, **test_metrics}


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import get_datasets

    # Load data
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=1)

    # Extract validation data if available
    val_X, val_y = val_pair if val_pair else (None, None)

    # Train model
    model, history = train_lstm(
        dev_X=dev_X,
        dev_y=dev_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        config={
            "lstm_units": 64,
            "dense_units": 32,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        },
    )

