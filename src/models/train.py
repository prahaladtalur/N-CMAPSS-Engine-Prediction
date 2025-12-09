"""
Training pipeline for RUL prediction models with wandb integration.

Supports easy model switching via the model registry.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import wandb
from wandb.integration.keras import WandbCallback

from src.models.architectures import ModelRegistry, get_model, list_available_models
from src.utils.metrics import compute_all_metrics, format_metrics


def prepare_sequences(
    X: List[np.ndarray],
    y: List[np.ndarray],
    max_sequence_length: Optional[int] = None,
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


def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        X_train: Training features (samples, timesteps, features)
        X_val: Validation features (optional)
        X_test: Test features (optional)

    Returns:
        Normalized arrays and fitted scaler
    """
    n_samples, n_timesteps, n_features = X_train.shape

    # Reshape for scaler
    X_train_flat = X_train.reshape(-1, n_features)
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_norm = X_train_flat.reshape(n_samples, n_timesteps, n_features)

    X_val_norm = None
    X_test_norm = None

    if X_val is not None:
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_flat = scaler.transform(X_val_flat)
        X_val_norm = X_val_flat.reshape(X_val.shape[0], n_timesteps, n_features)

    if X_test is not None:
        X_test_flat = X_test.reshape(-1, n_features)
        X_test_flat = scaler.transform(X_test_flat)
        X_test_norm = X_test_flat.reshape(X_test.shape[0], n_timesteps, n_features)

    return X_train_norm, X_val_norm, X_test_norm, scaler


def train_model(
    dev_X: List[np.ndarray],
    dev_y: List[np.ndarray],
    model_name: str = "lstm",
    val_X: Optional[List[np.ndarray]] = None,
    val_y: Optional[List[np.ndarray]] = None,
    test_X: Optional[List[np.ndarray]] = None,
    test_y: Optional[List[np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    project_name: str = "n-cmapss-rul-prediction",
    run_name: Optional[str] = None,
    normalize: bool = True,
    visualize: bool = True,
) -> Tuple[keras.Model, Dict[str, Any], Dict[str, float]]:
    """
    Train a model with wandb logging and comprehensive evaluation.

    Args:
        dev_X: Development set features
        dev_y: Development set labels
        model_name: Name of model architecture (see list_available_models())
        val_X: Validation set features (optional)
        val_y: Validation set labels (optional)
        test_X: Test set features (optional, for final evaluation)
        test_y: Test set labels (optional, for final evaluation)
        config: Training configuration dictionary
        project_name: Wandb project name
        run_name: Wandb run name (defaults to model_name)
        normalize: Whether to normalize input features
        visualize: Whether to generate visualizations

    Returns:
        Tuple of (trained_model, training_history, test_metrics)
    """
    # Default configuration
    default_config = {
        "units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "max_sequence_length": None,
        "validation_split": 0.2,
        "patience_early_stop": 10,
        "patience_lr_reduce": 5,
    }

    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}

    # Set run name
    if run_name is None:
        run_name = f"{model_name}-run"

    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={**config, "model_name": model_name},
        reinit=True,
    )

    # Prepare training data
    print(f"\nPreparing training data for {model_name}...")
    X_train, y_train = prepare_sequences(dev_X, dev_y, config["max_sequence_length"])
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Prepare validation data
    X_val, y_val = None, None
    if val_X is not None and val_y is not None:
        X_val, y_val = prepare_sequences(val_X, val_y, config["max_sequence_length"])
        print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    # Prepare test data
    X_test, y_test = None, None
    if test_X is not None and test_y is not None:
        X_test, y_test = prepare_sequences(test_X, test_y, config["max_sequence_length"])
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Normalize if requested
    scaler = None
    if normalize:
        print("Normalizing features...")
        X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    print(f"\nBuilding {model_name} model...")
    loss_func = config.get("loss", "mse")
    model = get_model(
        model_name,
        input_shape=input_shape,
        units=config["units"],
        dense_units=config["dense_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
        loss=loss_func,
    )

    # Print loss function info
    loss_name = loss_func if isinstance(loss_func, str) else loss_func.__name__
    print(f"Loss function: {loss_name}")

    print("\nModel Architecture:")
    model.summary()

    # Log model info to wandb
    wandb.config.update(
        {
            "input_shape": input_shape,
            "model_params": model.count_params(),
            "available_models": list_available_models(),
        }
    )

    # Prepare validation data for training
    validation_data = (X_val, y_val) if X_val is not None else None

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
            patience=config["patience_early_stop"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=config["patience_lr_reduce"],
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

    # Evaluate on test set
    test_metrics = {}
    y_pred = None
    if X_test is not None and y_test is not None:
        print("\nEvaluating on test set...")
        y_pred = model.predict(X_test, verbose=0).flatten()
        test_metrics = compute_all_metrics(y_test, y_pred)

        print(format_metrics(test_metrics))

        # Log all metrics to wandb
        wandb_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        wandb.log(wandb_metrics)

        # Generate visualizations if requested
        if visualize:
            from src.utils.training_viz import (
                plot_training_history,
                plot_predictions,
                plot_error_distribution,
            )

            print("\nGenerating visualizations...")

            # Create results directory
            results_dir = f"results/{run_name}"
            os.makedirs(results_dir, exist_ok=True)

            # Plot training history
            plot_training_history(
                history.history,
                model_name,
                save_path=f"{results_dir}/training_history.png",
            )

            # Plot predictions
            plot_predictions(
                y_test,
                y_pred,
                model_name,
                save_path=f"{results_dir}/predictions.png",
            )

            # Plot error distribution
            plot_error_distribution(
                y_test,
                y_pred,
                model_name,
                save_path=f"{results_dir}/error_distribution.png",
            )

            # Log plots to wandb
            wandb.log(
                {
                    "training_history": wandb.Image(f"{results_dir}/training_history.png"),
                    "predictions": wandb.Image(f"{results_dir}/predictions.png"),
                    "error_distribution": wandb.Image(f"{results_dir}/error_distribution.png"),
                }
            )

    # Log final training metrics
    final_metrics = {
        "final_train_loss": history.history["loss"][-1],
        "final_train_mae": history.history["mae"][-1],
        "epochs_trained": len(history.history["loss"]),
    }
    if validation_data:
        final_metrics["final_val_loss"] = history.history["val_loss"][-1]
        final_metrics["final_val_mae"] = history.history["val_mae"][-1]

    wandb.log(final_metrics)
    wandb.finish()

    return model, history.history, test_metrics


def compare_models(
    dev_X: List[np.ndarray],
    dev_y: List[np.ndarray],
    model_names: List[str] = None,
    val_X: Optional[List[np.ndarray]] = None,
    val_y: Optional[List[np.ndarray]] = None,
    test_X: Optional[List[np.ndarray]] = None,
    test_y: Optional[List[np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    project_name: str = "n-cmapss-rul-comparison",
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same dataset.

    Args:
        dev_X: Development set features
        dev_y: Development set labels
        model_names: List of model names to compare (default: all available)
        val_X: Validation set features
        val_y: Validation set labels
        test_X: Test set features
        test_y: Test set labels
        config: Training configuration
        project_name: Wandb project name

    Returns:
        Dictionary with results for each model
    """
    if model_names is None:
        model_names = list_available_models()

    print(f"\nComparing {len(model_names)} models: {model_names}")
    print("=" * 60)

    results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print("=" * 60)

        try:
            model, history, test_metrics = train_model(
                dev_X=dev_X,
                dev_y=dev_y,
                model_name=model_name,
                val_X=val_X,
                val_y=val_y,
                test_X=test_X,
                test_y=test_y,
                config=config,
                project_name=project_name,
                run_name=f"{model_name}-comparison",
                visualize=False,  # Skip individual visualizations
            )

            results[model_name] = {
                "model": model,
                "history": history,
                "metrics": test_metrics,
            }

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Generate comparison visualization
    if test_X is not None:
        from src.utils.training_viz import plot_model_comparison
        from src.utils.metrics import compare_models as find_best_model

        metrics_only = {
            name: res["metrics"]
            for name, res in results.items()
            if "metrics" in res and res["metrics"]
        }

        if metrics_only:
            print("\n" + "=" * 60)
            print("Model Comparison Results")
            print("=" * 60)

            # Find best model
            best_model, best_metrics = find_best_model(metrics_only, "rmse")
            print(f"\nBest Model (by RMSE): {best_model}")
            print(format_metrics(best_metrics))

            # Plot comparison
            os.makedirs("results/comparison", exist_ok=True)
            plot_model_comparison(
                metrics_only,
                save_path="results/comparison/model_comparison.png",
            )

    return results


# Legacy function for backwards compatibility
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
    Legacy function - trains LSTM model.
    Use train_model() with model_name parameter for new code.
    """
    # Map old config keys to new ones
    if config:
        if "lstm_units" in config:
            config["units"] = config.pop("lstm_units")

    model, history, test_metrics = train_model(
        dev_X=dev_X,
        dev_y=dev_y,
        model_name="lstm",
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        config=config,
        project_name=project_name,
        run_name=run_name,
    )

    # Merge test metrics into history for backwards compatibility
    combined_history = {**history, **test_metrics}
    return model, combined_history


if __name__ == "__main__":
    from src.data.load_data import get_datasets

    print("Available models:", list_available_models())

    # Load data
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=1)
    val_X, val_y = val_pair if val_pair else (None, None)

    # Train a single model
    model, history, metrics = train_model(
        dev_X=dev_X,
        dev_y=dev_y,
        model_name="attention_lstm",  # Try SOTA attention LSTM
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        config={"epochs": 30},
    )
