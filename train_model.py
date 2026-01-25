#!/usr/bin/env python
"""
Easy CLI for training and comparing RUL prediction models.

Usage examples:
    # Train a single model
    python train_model.py --model lstm

    # Train with custom config
    python train_model.py --model transformer --epochs 100 --units 128

    # Train using a JSON config file (like gru_quick_search.json)
    python train_model.py --config sweeps/gru_quick_search.json

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
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import wandb
from wandb.integration.keras import WandbCallback

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv is optional

from src.data.load_data import get_datasets
from src.models.architectures import (
    ModelRegistry,
    get_model,
    list_available_models,
    get_model_info,
    get_model_recommendations,
)
from src.utils.metrics import compute_all_metrics, format_metrics
from src.search import run_hparam_search


# ============================================================================
# Training Functions (moved from src/models/train.py)
# ============================================================================


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
    use_early_stop: bool = True,
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
        "use_early_stop": True,
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
    model = get_model(
        model_name,
        input_shape=input_shape,
        units=config["units"],
        dense_units=config["dense_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )

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
            log_weights=False,
            log_gradients=False,
            save_model=False,
            save_graph=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=config["patience_lr_reduce"],
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    if use_early_stop:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=config["patience_early_stop"],
                restore_best_weights=True,
                verbose=1,
            )
        )

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

    use_early_stop = True
    if config and "use_early_stop" in config:
        use_early_stop = bool(config["use_early_stop"])

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
                use_early_stop=use_early_stop,
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

    .. deprecated:: 0.1.0
        Use :func:`train_model` with ``model_name="lstm"`` instead.

    This function is kept for backwards compatibility but will be removed
    in a future version. Migrate to:

    .. code-block:: python

        model, history, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name="lstm",
            ...
        )
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


# ============================================================================
# JSON Config Loading
# ============================================================================


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from a JSON file.

    Supports both simple config files and wandb sweep config files.
    For sweep configs, uses the first parameter combination (or base_config if no parameters).

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary with model_name, config, fd, project_name, normalize, visualize
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if this is a sweep config (has 'parameters' or 'method')
    is_sweep_config = "parameters" in data or "method" in data

    if is_sweep_config:
        # Extract model name
        model_name = data.get("model", "lstm")

        # Get base config
        base_config = data.get("base_config", {})

        # If there are parameters, use the first combination
        # For grid search, this would be the first combination
        # For random search, we'll just use base_config
        if "parameters" in data and data["parameters"]:
            # For a single training run, we'll use the first value of each parameter
            # or just use base_config if user wants a full sweep, they should use --search-config
            param_values = {}
            for key, values in data["parameters"].items():
                if isinstance(values, list) and len(values) > 0:
                    param_values[key] = values[0]  # Use first value
            config = {**base_config, **param_values}
        else:
            config = base_config

        # Extract other settings
        result = {
            "model_name": model_name,
            "config": config,
            "fd": data.get("fd", 1),
            "project_name": data.get("project_name", "n-cmapss-rul-prediction"),
            "normalize": data.get("normalize", True),
            "visualize": data.get("visualize", True),
        }
    else:
        # Simple config file - expect model_name and config directly
        result = {
            "model_name": data.get("model_name", data.get("model", "lstm")),
            "config": data.get("config", {}),
            "fd": data.get("fd", 1),
            "project_name": data.get("project_name", "n-cmapss-rul-prediction"),
            "normalize": data.get("normalize", True),
            "visualize": data.get("visualize", True),
        }

    return result


# ============================================================================
# CLI Functions
# ============================================================================


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

  # Train using a JSON config file
  python train_model.py --config sweeps/gru_quick_search.json

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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (overrides other training args)",
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
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument(
        "--patience-early-stop",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--patience-lr-reduce",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (default: 5)",
    )
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
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping to run all epochs",
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
    parser.add_argument(
        "--search-config",
        type=str,
        default=None,
        help="Path to JSON hyperparameter search specification",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=1,
        help="Initial number of parallel search workers",
    )
    parser.add_argument(
        "--search-max-workers",
        type=int,
        default=None,
        help="Upper bound for worker pool (defaults to CPU count)",
    )
    parser.add_argument(
        "--search-throttle-file",
        type=str,
        default=None,
        help="Optional file containing an integer worker count that can be edited live",
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_models:
        print_models_info()
        return

    if args.recommend:
        print_recommendations()
        return

    if args.search_config:
        run_hparam_search(args)
        return

    # Handle JSON config file
    if args.config:
        try:
            json_config = load_config_from_json(args.config)
            print(f"\nLoaded configuration from: {args.config}")
            print(f"Model: {json_config['model_name']}")
            print(f"FD: {json_config['fd']}")
            print(f"Config: {json_config['config']}")

            # Override args with JSON config values
            args.model = json_config["model_name"]
            args.fd = json_config["fd"]
            args.project = json_config["project_name"]
            args.no_normalize = not json_config["normalize"]
            args.no_visualize = not json_config["visualize"]

            # Merge JSON config into training config
            json_training_config = json_config["config"]
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        json_training_config = {}

    # Validate arguments
    if not args.compare and not args.compare_all and not args.model:
        parser.print_help()
        print("\n\nError: Must specify --model, --config, --compare, or --compare-all")
        sys.exit(1)

    # Load data
    print(f"\nLoading N-CMAPSS FD{args.fd} dataset...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=args.fd)
    val_X, val_y = val_pair if val_pair else (None, None)

    # Training configuration - JSON config overrides CLI args
    config = {
        "units": json_training_config.get("units", args.units),
        "dense_units": json_training_config.get("dense_units", args.dense_units),
        "dropout_rate": json_training_config.get("dropout_rate", args.dropout),
        "learning_rate": json_training_config.get("learning_rate", args.lr),
        "batch_size": json_training_config.get("batch_size", args.batch_size),
        "epochs": json_training_config.get("epochs", args.epochs),
        "max_sequence_length": json_training_config.get("max_sequence_length", args.max_seq_length),
        "patience_early_stop": json_training_config.get(
            "patience_early_stop", args.patience_early_stop
        ),
        "patience_lr_reduce": json_training_config.get(
            "patience_lr_reduce", args.patience_lr_reduce
        ),
        "use_early_stop": json_training_config.get("use_early_stop", not args.no_early_stop),
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
            use_early_stop=config.get("use_early_stop", True),
        )

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE: {model_name}")
        print("=" * 80)

        if metrics:
            print("\nTest Set Metrics:")
            print(format_metrics(metrics))

        run_name = args.run_name or f"{model_name}-run"
        print(f"\nResults saved to: results/{run_name}/")
        print(f"Check wandb project: {args.project}")


if __name__ == "__main__":
    main()
