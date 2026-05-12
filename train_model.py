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

    # Train with a fixed seed (reproducible)
    python train_model.py --model mstcn --seed 42

    # Multi-seed experiment (reports mean ± std)
    python train_model.py --model mstcn --seed 42 --num-seeds 5
"""

import argparse
import inspect
import json
import os
import random
import subprocess
import sys
import pickle
from datetime import datetime
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
    compile_model_for_training,
    get_model,
    list_available_models,
    get_model_info,
    get_model_recommendations,
)
from src.utils.metrics import compute_all_metrics, format_metrics
from src.search import run_hparam_search

# ============================================================================
# Reproducibility Utilities
# ============================================================================


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for full experiment reproducibility.

    NOTE: PYTHONHASHSEED must be set before Python starts to fully take effect.
    Set it in your shell: ``export PYTHONHASHSEED=42`` before running training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_git_hash() -> str:
    """Return the current git commit hash for experiment traceability."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


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


def clip_rul_targets(y: np.ndarray, max_rul: Optional[float] = None) -> np.ndarray:
    """Cap RUL targets at ``max_rul`` when configured."""
    clipped = np.asarray(y, dtype=np.float32).copy()
    if max_rul is not None:
        clipped = np.minimum(clipped, max_rul)
    return clipped


def transform_targets(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    *,
    clip_value: Optional[float] = None,
    scaling: str = "none",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Apply issue-25 target preprocessing experiments in a reversible way."""
    y_train_metric = clip_rul_targets(y_train, clip_value)
    y_val_metric = clip_rul_targets(y_val, clip_value) if y_val is not None else None
    y_test_metric = clip_rul_targets(y_test, clip_value) if y_test is not None else None

    metadata = {
        "clip_value": float(clip_value) if clip_value is not None else None,
        "scaling": scaling,
        "scale_min": float(y_train_metric.min()) if y_train_metric.size else 0.0,
        "scale_max": float(y_train_metric.max()) if y_train_metric.size else 0.0,
    }

    if scaling == "none":
        return y_train_metric, y_val_metric, y_test_metric, metadata

    if scaling != "minmax":
        raise ValueError("Unsupported target scaling. Available: none, minmax")

    y_range = metadata["scale_max"] - metadata["scale_min"]
    if y_range == 0:
        zeros = np.zeros_like(y_train_metric, dtype=np.float32)
        y_val_scaled = (
            np.zeros_like(y_val_metric, dtype=np.float32) if y_val_metric is not None else None
        )
        y_test_scaled = (
            np.zeros_like(y_test_metric, dtype=np.float32) if y_test_metric is not None else None
        )
        return zeros, y_val_scaled, y_test_scaled, metadata

    def _scale(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        return ((values - metadata["scale_min"]) / y_range).astype(np.float32)

    return _scale(y_train_metric), _scale(y_val_metric), _scale(y_test_metric), metadata


def inverse_transform_targets(y: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """Map model outputs back into the clipped/raw RUL space used for evaluation."""
    values = np.asarray(y, dtype=np.float32)
    if metadata.get("scaling") != "minmax":
        return values

    scale_min = metadata["scale_min"]
    scale_max = metadata["scale_max"]
    return values * (scale_max - scale_min) + scale_min


def save_model_checkpoint(
    model: keras.Model,
    scaler: Optional[StandardScaler],
    config: Dict[str, Any],
    test_metrics: Dict[str, float],
    model_name: str,
    run_name: str,
    y_min: float,
    y_max: float,
    target_transform: Optional[Dict[str, Any]] = None,
    prediction_calibrator: Optional[Dict[str, Any]] = None,
    save_dir: str = "models",
) -> str:
    """
    Save trained model with all necessary artifacts for inference.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler (or None if no normalization)
        config: Training configuration dictionary
        test_metrics: Test set evaluation metrics
        model_name: Name of model architecture
        run_name: Name of training run
        y_min: Minimum RUL value (for normalization)
        y_max: Maximum RUL value (for normalization)
        save_dir: Base directory for saving models

    Returns:
        Path to saved model directory
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(save_dir) / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Saving model checkpoint to: {model_dir}")
    print(f"{'='*60}")

    # 1. Save model in Keras format
    model_path = model_dir / "model.keras"
    model.save(model_path)
    print(f"✓ Model saved: {model_path.name}")

    # 2. Save scaler (if used)
    if scaler is not None:
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✓ Scaler saved: {scaler_path.name}")

    # 3. Save RUL normalization parameters
    rul_scaler_path = model_dir / "rul_scaler.json"
    with open(rul_scaler_path, "w") as f:
        json.dump(
            {
                "y_min": float(y_min),
                "y_max": float(y_max),
                "clip_value": (
                    None if target_transform is None else target_transform.get("clip_value")
                ),
                "target_scaling": (
                    None if target_transform is None else target_transform.get("scaling")
                ),
                "scale_min": (
                    None if target_transform is None else target_transform.get("scale_min")
                ),
                "scale_max": (
                    None if target_transform is None else target_transform.get("scale_max")
                ),
                "prediction_clip": config.get("prediction_clip", "none"),
                "prediction_calibration": prediction_calibrator or {"method": "none"},
            },
            f,
            indent=2,
        )
    print(f"✓ RUL scaler saved: {rul_scaler_path.name}")

    # 4. Save training configuration
    config_path = model_dir / "config.json"
    # Make config JSON-serializable
    safe_config = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in config.items()}
    with open(config_path, "w") as f:
        json.dump(safe_config, f, indent=2)
    print(f"✓ Config saved: {config_path.name}")

    # 5. Save test metrics
    metrics_path = model_dir / "metrics.json"
    # Make metrics JSON-serializable
    safe_metrics = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in test_metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(safe_metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path.name}")

    # 6. Save model metadata
    metadata = {
        "model_name": model_name,
        "run_name": run_name,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "architecture": {
            "num_parameters": int(model.count_params()),
            "num_layers": len(model.layers),
            "input_shape": [int(x) for x in model.input_shape[1:]],
        },
        "performance": {
            "rmse": safe_metrics.get("rmse"),
            "mae": safe_metrics.get("mae"),
            "r2": safe_metrics.get("r2"),
            "rmse_normalized": safe_metrics.get("rmse_normalized"),
            "accuracy_at_20": safe_metrics.get("accuracy_20"),
        },
        "training": {
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "max_sequence_length": config.get("max_sequence_length"),
            "loss_name": config.get("loss_name"),
            "loss_alpha": config.get("loss_alpha"),
            "rul_clip_value": config.get("rul_clip_value"),
            "target_scaling": config.get("target_scaling"),
            "prediction_clip": config.get("prediction_clip"),
            "calibration_method": config.get("calibration_method"),
            "sample_weighting": config.get("sample_weighting"),
        },
        "files": {
            "model": "model.keras",
            "scaler": "scaler.pkl" if scaler is not None else None,
            "config": "config.json",
            "metrics": "metrics.json",
            "rul_scaler": "rul_scaler.json",
        },
        "usage": {
            "load_model": f'model = keras.models.load_model("{model_path}")',
            "load_scaler": (
                f'scaler = pickle.load(open("{model_dir}/scaler.pkl", "rb"))' if scaler else None
            ),
            "predict": "from predict import RULPredictor; predictor = RULPredictor(model_path='...')",
        },
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path.name}")

    # 7. Create README for the model
    readme_content = f"""# {model_name.upper()} Model Checkpoint

**Saved**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Run Name**: {run_name}

## Performance

| Metric | Value |
|--------|-------|
| RMSE | {safe_metrics.get('rmse', 'N/A'):.2f} cycles |
| MAE | {safe_metrics.get('mae', 'N/A'):.2f} cycles |
| R² Score | {safe_metrics.get('r2', 'N/A'):.4f} |
| Accuracy@20 | {safe_metrics.get('accuracy_20', 'N/A'):.2f}% |

## Usage

### Python API

```python
from predict import RULPredictor
import numpy as np

# Load predictor
predictor = RULPredictor(model_path='{model_path}')

# Make prediction
sensor_data = np.load('your_data.npy')  # Shape: (timesteps, 32)
result = predictor.predict_single(sensor_data)

print(f"Predicted RUL: {{result['prediction']:.2f}} cycles")
```

### Command Line

```bash
python predict.py --model-path {model_path} --input-file your_data.npy
```

## Files

- `model.keras` - Trained Keras model
- `scaler.pkl` - Feature normalization scaler
- `config.json` - Training configuration
- `metrics.json` - Test set evaluation metrics
- `rul_scaler.json` - RUL normalization parameters
- `metadata.json` - Complete model metadata

## Configuration

{json.dumps(safe_config, indent=2)}
"""

    readme_path = model_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"✓ README saved: {readme_path.name}")

    print(f"{'='*60}")
    print(f"✅ Model checkpoint saved successfully!")
    print(f"📂 Location: {model_dir}")
    print(f"{'='*60}\n")

    return str(model_dir)


def make_json_safe(value):
    """Recursively convert NumPy scalars and containers to JSON-safe values."""
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


BEST_ACCURACY_RECIPE = {
    # 1) Use the proven short degradation window from the benchmark.
    "max_sequence_length": 1000,
    # 2) Use the best tuned architecture family with a slightly richer head.
    "units": 64,
    "dense_units": 64,
    "dropout_rate": 0.18,
    "model_kwargs": {"dilation_rates": [1, 2, 4, 8, 16], "pooling": "attention"},
    # 3) Use the strongest historical optimizer settings from local tuning.
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    # 4) Stabilize the RUL objective.
    "loss_name": "asymmetric_huber",
    "loss_alpha": 1.5,
    "loss_delta": 0.08,
    "target_scaling": "minmax",
    # 5) Keep paper-style target clipping configurable but enabled for the recipe.
    "rul_clip_value": 125,
    # 6) Make the optimizer less sensitive to large updates.
    "optimizer_name": "adamw",
    "weight_decay": 1e-5,
    "gradient_clipnorm": 1.0,
    # 7) Let validation MAE drive the saved/restored best epoch.
    "monitor_metric": "val_rmse",
    "monitor_mode": "min",
    "early_stop_min_delta": 1e-4,
    "patience_early_stop": 18,
    "patience_lr_reduce": 7,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-7,
    # 8) Spend more training signal near failure where accuracy matters most.
    "sample_weighting": "low_rul",
    "sample_weight_strength": 1.0,
    "sample_weight_power": 2.0,
    "sample_weight_min": 0.5,
    "sample_weight_max": 2.5,
    # 9) Prevent physically invalid or out-of-range predictions at evaluation.
    "prediction_clip": "train_range",
    # 10) Calibrate final predictions on the validation split without test leakage.
    "calibration_method": "linear",
}


def get_accuracy_recipe_config(recipe_name: str) -> Dict[str, Any]:
    """Return a named accuracy recipe config."""
    if recipe_name == "none":
        return {}
    if recipe_name != "best":
        raise ValueError("Unsupported accuracy recipe. Available: none, best")
    return dict(BEST_ACCURACY_RECIPE)


def compute_sample_weights(
    y_values: np.ndarray,
    *,
    mode: str = "none",
    strength: float = 1.0,
    power: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    num_bins: int = 10,
) -> Optional[np.ndarray]:
    """Compute optional per-sample weights for accuracy-focused training."""
    if mode == "none":
        return None

    y_values = np.asarray(y_values, dtype=np.float32)
    if y_values.size == 0:
        return None

    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_range = max(y_max - y_min, 1e-8)
    normalized = np.clip((y_values - y_min) / y_range, 0.0, 1.0)

    if mode == "low_rul":
        weights = 1.0 + strength * np.power(1.0 - normalized, power)
    elif mode == "inverse_rul":
        weights = np.power(1.0 / np.maximum(normalized, 0.05), power)
        weights = 1.0 + strength * (weights / np.mean(weights) - 1.0)
    elif mode == "balanced_bins":
        bin_count = max(2, min(num_bins, y_values.size))
        edges = np.linspace(y_min, y_max, bin_count + 1)
        bin_ids = np.clip(np.digitize(y_values, edges[1:-1]), 0, bin_count - 1)
        counts = np.bincount(bin_ids, minlength=bin_count).astype(np.float32)
        weights = 1.0 / np.maximum(counts[bin_ids], 1.0)
        weights = weights / np.mean(weights)
    else:
        raise ValueError(
            "Unsupported sample weighting. Available: none, low_rul, inverse_rul, balanced_bins"
        )

    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / max(float(np.mean(weights)), 1e-8)
    return weights.astype(np.float32)


def apply_prediction_postprocessing(
    y_pred: np.ndarray,
    *,
    clip_mode: str = "none",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> np.ndarray:
    """Apply simple physical constraints to RUL predictions."""
    values = np.asarray(y_pred, dtype=np.float32)
    if clip_mode == "none":
        return values
    if clip_mode == "nonnegative":
        return np.maximum(values, 0.0)
    if clip_mode == "train_range":
        if y_min is None or y_max is None:
            return np.maximum(values, 0.0)
        return np.clip(values, y_min, y_max)
    raise ValueError("Unsupported prediction clip mode. Available: none, nonnegative, train_range")


def fit_prediction_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str = "none",
) -> Dict[str, Any]:
    """Fit a validation-only prediction calibration transform."""
    if method == "none":
        return {"method": "none"}

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size == 0 or y_pred.size == 0:
        return {"method": "none"}

    if method == "bias":
        return {"method": "bias", "offset": float(np.mean(y_true - y_pred))}
    if method == "linear":
        if np.std(y_pred) < 1e-8:
            return {"method": "bias", "offset": float(np.mean(y_true - y_pred))}
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        return {"method": "linear", "slope": float(slope), "intercept": float(intercept)}

    raise ValueError("Unsupported calibration method. Available: none, bias, linear")


def apply_prediction_calibrator(y_pred: np.ndarray, calibrator: Dict[str, Any]) -> np.ndarray:
    """Apply a fitted prediction calibrator."""
    values = np.asarray(y_pred, dtype=np.float32)
    method = calibrator.get("method", "none")
    if method == "none":
        return values
    if method == "bias":
        return values + float(calibrator.get("offset", 0.0))
    if method == "linear":
        return values * float(calibrator.get("slope", 1.0)) + float(
            calibrator.get("intercept", 0.0)
        )
    raise ValueError(f"Unsupported calibration method in fitted calibrator: {method}")


def should_log_sota_gap(config: Dict[str, Any]) -> bool:
    """Only log paper SOTA gaps when dataset identity is explicitly comparable."""
    return config.get("dataset", "ncmapss") == config.get("sota_target_dataset", "cmapss")


def add_sota_summary_metrics(
    summary_data: Dict[str, Any],
    test_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> None:
    """Add normalized metric summaries without inventing cross-dataset SOTA gaps."""
    if "rmse_normalized" not in test_metrics:
        return

    summary_data.update(
        {
            "results/best_rmse_normalized": float(test_metrics["rmse_normalized"]),
            "results/best_mae_normalized": float(test_metrics["mae_normalized"]),
        }
    )

    if should_log_sota_gap(config):
        summary_data.update(
            {
                "results/rmse_norm_gap_vs_sota": float(test_metrics["rmse_normalized"] / 0.032),
                "results/mae_norm_gap_vs_sota": float(test_metrics["mae_normalized"] / 0.026),
            }
        )
    else:
        summary_data.update(
            {
                "results/sota_gap_skipped": 1,
                "results/sota_target_dataset": config.get("sota_target_dataset", "cmapss"),
                "results/current_dataset": config.get("dataset", "ncmapss"),
            }
        )


class RULMetricCallback(keras.callbacks.Callback):
    """Log validation metrics in cycle-space RUL units."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_val_true: np.ndarray,
        target_transform: Dict[str, Any],
        *,
        clip_value: Optional[float],
        prediction_clip: str,
        y_min: float,
        y_max: float,
    ):
        super().__init__()
        self.X_val = X_val
        self.y_val_true = clip_rul_targets(y_val_true, clip_value)
        self.target_transform = target_transform
        self.prediction_clip = prediction_clip
        self.y_min = y_min
        self.y_max = y_max

    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        logs = logs or {}
        y_pred_fit = self.model.predict(self.X_val, verbose=0).flatten()
        y_pred = inverse_transform_targets(y_pred_fit, self.target_transform)
        y_pred = apply_prediction_postprocessing(
            y_pred,
            clip_mode=self.prediction_clip,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        metrics = compute_all_metrics(self.y_val_true, y_pred, y_min=self.y_min, y_max=self.y_max)
        logs["val_rmse"] = metrics["rmse"]
        logs["val_rul_mae"] = metrics["mae"]
        logs["val_accuracy_10"] = metrics["accuracy_10"]
        logs["val_accuracy_15"] = metrics["accuracy_15"]
        logs["val_accuracy_20"] = metrics["accuracy_20"]
        logs["val_phm_score_normalized"] = metrics["phm_score_normalized"]


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
    save_checkpoint: bool = True,
    seed: int = 42,
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
        "loss_name": "asymmetric_mse",
        "loss_alpha": 2.0,
        "loss_delta": 1.0,
        "rul_clip_value": None,
        "target_scaling": "none",
        "gradient_clipnorm": None,
        "gradient_clipvalue": None,
        "optimizer_name": "adam",
        "weight_decay": None,
        "validation_split": 0.2,
        "patience_early_stop": 10,
        "patience_lr_reduce": 5,
        "lr_reduce_factor": 0.5,
        "min_lr": 1e-7,
        "monitor_metric": "auto",
        "monitor_mode": "auto",
        "early_stop_min_delta": 0.0,
        "use_early_stop": True,
        "sample_weighting": "none",
        "sample_weight_strength": 1.0,
        "sample_weight_power": 1.0,
        "sample_weight_min": 0.25,
        "sample_weight_max": 4.0,
        "prediction_clip": "none",
        "calibration_method": "none",
        "model_kwargs": {},
        "shuffle": True,
        "dataset": "ncmapss",
        "sota_target_dataset": "cmapss",
        "fixed_metric_max_rul": None,
    }

    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}

    # Inject seed into config so it gets saved with all artifacts
    config["seed"] = seed

    # Set random seeds for reproducibility
    set_seeds(seed)

    # Capture git hash for traceability
    git_hash = get_git_hash()

    # Set run name
    if run_name is None:
        run_name = f"{model_name}-run"

    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={**config, "model_name": model_name, "git_hash": git_hash},
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

    y_train_fit, y_val_fit, y_test_fit, target_transform = transform_targets(
        y_train,
        y_val,
        y_test,
        clip_value=config["rul_clip_value"],
        scaling=config["target_scaling"],
    )

    # Calculate RUL normalization range in the evaluation space.
    y_min = float(target_transform["scale_min"])
    y_max = float(target_transform["scale_max"])
    print(f"RUL range: [{y_min:.2f}, {y_max:.2f}] cycles")
    if config["rul_clip_value"] is not None:
        print(f"RUL clipping enabled at: {config['rul_clip_value']:.2f} cycles")
    if config["target_scaling"] != "none":
        print(f"Target scaling enabled: {config['target_scaling']}")

    # Normalize if requested
    scaler = None
    if normalize:
        print("Normalizing features...")
        X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    print(f"\nBuilding {model_name} model...")
    model_kwargs = config.get("model_kwargs", {})
    if model_kwargs:
        build_signature = inspect.signature(ModelRegistry.get(model_name).build)
        accepted_kwargs = set(build_signature.parameters)
        filtered_model_kwargs = {
            key: value for key, value in model_kwargs.items() if key in accepted_kwargs
        }
        ignored_kwargs = sorted(set(model_kwargs) - set(filtered_model_kwargs))
        if ignored_kwargs:
            print(
                f"Ignoring unsupported model kwargs for {model_name}: "
                f"{', '.join(ignored_kwargs)}"
            )
    else:
        filtered_model_kwargs = {}

    model = get_model(
        model_name,
        input_shape=input_shape,
        units=config["units"],
        dense_units=config["dense_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
        **filtered_model_kwargs,
    )
    model = compile_model_for_training(
        model,
        learning_rate=config["learning_rate"],
        loss_name=config["loss_name"],
        loss_alpha=config["loss_alpha"],
        loss_delta=config["loss_delta"],
        gradient_clipnorm=config["gradient_clipnorm"],
        gradient_clipvalue=config["gradient_clipvalue"],
        optimizer_name=config["optimizer_name"],
        weight_decay=config["weight_decay"],
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

    sample_weights = compute_sample_weights(
        clip_rul_targets(y_train, config["rul_clip_value"]),
        mode=config["sample_weighting"],
        strength=config["sample_weight_strength"],
        power=config["sample_weight_power"],
        min_weight=config["sample_weight_min"],
        max_weight=config["sample_weight_max"],
    )
    if sample_weights is not None:
        print(
            "Sample weighting enabled: "
            f"{config['sample_weighting']} "
            f"(min={sample_weights.min():.3f}, max={sample_weights.max():.3f})"
        )

    # Prepare validation data for training
    validation_data = (X_val, y_val_fit) if X_val is not None and y_val_fit is not None else None
    monitor_metric = config["monitor_metric"]
    if monitor_metric == "auto":
        monitor_metric = "val_loss" if validation_data else "loss"
    if validation_data is None and monitor_metric.startswith("val_"):
        print(
            f"Validation metric '{config['monitor_metric']}' requested without a validation "
            "split; falling back to loss."
        )
        monitor_metric = "loss"

    # Callbacks
    callbacks = []
    if validation_data:
        callbacks.append(
            RULMetricCallback(
                X_val,
                y_val,
                target_transform,
                clip_value=config["rul_clip_value"],
                prediction_clip=config["prediction_clip"],
                y_min=y_min,
                y_max=y_max,
            )
        )
    callbacks.append(
        WandbCallback(
            monitor=monitor_metric,
            log_weights=False,
            log_gradients=False,
            save_model=False,
            save_graph=False,
        )
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=config["monitor_mode"],
            factor=config["lr_reduce_factor"],
            patience=config["patience_lr_reduce"],
            min_lr=config["min_lr"],
            verbose=1,
        )
    )
    if use_early_stop:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode=config["monitor_mode"],
                patience=config["patience_early_stop"],
                min_delta=config["early_stop_min_delta"],
                restore_best_weights=True,
                verbose=1,
            )
        )

    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train_fit,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_val, y_val_fit) if X_val is not None and y_val_fit is not None else None,
        validation_split=config["validation_split"] if validation_data is None else None,
        sample_weight=sample_weights,
        callbacks=callbacks,
        shuffle=config["shuffle"],
        verbose=1,
    )

    prediction_calibrator = {"method": "none"}
    if config["calibration_method"] != "none" and X_val is not None and y_val is not None:
        print(f"\nFitting validation calibration: {config['calibration_method']}")
        y_val_pred_fit = model.predict(X_val, verbose=0).flatten()
        y_val_pred = inverse_transform_targets(y_val_pred_fit, target_transform)
        y_val_pred = apply_prediction_postprocessing(
            y_val_pred,
            clip_mode=config["prediction_clip"],
            y_min=y_min,
            y_max=y_max,
        )
        y_val_eval = clip_rul_targets(y_val, config["rul_clip_value"])
        prediction_calibrator = fit_prediction_calibrator(
            y_val_eval,
            y_val_pred,
            method=config["calibration_method"],
        )
        print(f"Calibration parameters: {prediction_calibrator}")

    # Evaluate on test set
    test_metrics = {}
    y_pred = None
    if X_test is not None and y_test is not None:
        print("\nEvaluating on test set...")
        y_pred_fit = model.predict(X_test, verbose=0).flatten()
        y_pred = inverse_transform_targets(y_pred_fit, target_transform)
        y_pred = apply_prediction_postprocessing(
            y_pred,
            clip_mode=config["prediction_clip"],
            y_min=y_min,
            y_max=y_max,
        )
        y_pred = apply_prediction_calibrator(y_pred, prediction_calibrator)
        y_pred = apply_prediction_postprocessing(
            y_pred,
            clip_mode=config["prediction_clip"],
            y_min=y_min,
            y_max=y_max,
        )
        y_test_eval = clip_rul_targets(y_test, config["rul_clip_value"])
        test_metrics = compute_all_metrics(
            y_test_eval,
            y_pred,
            y_min=y_min,
            y_max=y_max,
            max_rul=config["fixed_metric_max_rul"],
        )

        print(format_metrics(test_metrics))

        # Log all metrics to wandb with test/ prefix
        wandb_metrics = {f"test/{k}": v for k, v in test_metrics.items()}

        # SOTA benchmarks from the MDFA paper are intentionally not compared
        # against this N-CMAPSS run. Those reported normalized targets appear
        # to use a C-MAPSS-style benchmark/denominator, so logging a gap here
        # would imply dataset comparability that we have not established.
        sota_dataset = config.get("sota_target_dataset", "cmapss")
        current_dataset = config.get("dataset", "ncmapss")
        SOTA_TARGETS = {
            "rmse_normalized": 0.032,
            "mae_normalized": 0.026,
            "r2": 0.987,
        }

        if (
            should_log_sota_gap(config)
            and "rmse_normalized" in test_metrics
            and "mae_normalized" in test_metrics
        ):
            rmse_gap = test_metrics["rmse_normalized"] / SOTA_TARGETS["rmse_normalized"]
            mae_gap = test_metrics["mae_normalized"] / SOTA_TARGETS["mae_normalized"]
            r2_gap = SOTA_TARGETS["r2"] - test_metrics["r2"]

            wandb_metrics.update(
                {
                    "test/rmse_normalized_gap": rmse_gap,
                    "test/mae_normalized_gap": mae_gap,
                    "test/r2_gap": r2_gap,
                    "sota/rmse_normalized_target": SOTA_TARGETS["rmse_normalized"],
                    "sota/mae_normalized_target": SOTA_TARGETS["mae_normalized"],
                    "sota/r2_target": SOTA_TARGETS["r2"],
                }
            )

            print(f"\n{'='*60}")
            print("Gap from SOTA (MDFA paper):")
            print(
                f"  RMSE (normalized): {test_metrics['rmse_normalized']:.4f} vs {SOTA_TARGETS['rmse_normalized']:.4f} (gap: {rmse_gap:.2f}x)"
            )
            print(
                f"  MAE (normalized):  {test_metrics['mae_normalized']:.4f} vs {SOTA_TARGETS['mae_normalized']:.4f} (gap: {mae_gap:.2f}x)"
            )
            print(
                f"  R² Score:          {test_metrics['r2']:.4f} vs {SOTA_TARGETS['r2']:.4f} (gap: {r2_gap:.4f})"
            )
            print("=" * 60)
        else:
            wandb_metrics.update(
                {
                    "sota/comparison_skipped": 1,
                    "sota/target_dataset": sota_dataset,
                    "sota/current_dataset": current_dataset,
                }
            )
            print(f"\n{'='*60}")
            print("Skipping SOTA gap comparison")
            print(f"  Current dataset: {current_dataset}; target dataset: {sota_dataset}.")
            print(
                "  The MDFA normalized targets are not logged as a gap unless the "
                "dataset/denominator is explicitly comparable."
            )
            print("=" * 60)

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
                y_test_eval, y_pred, model_name, save_path=f"{results_dir}/predictions.png"
            )

            # Plot error distribution
            plot_error_distribution(
                y_test_eval,
                y_pred,
                model_name,
                save_path=f"{results_dir}/error_distribution.png",
            )

            # Log plots to wandb
            wandb.log(
                {
                    "charts/training_history": wandb.Image(f"{results_dir}/training_history.png"),
                    "charts/predictions": wandb.Image(f"{results_dir}/predictions.png"),
                    "charts/error_distribution": wandb.Image(
                        f"{results_dir}/error_distribution.png"
                    ),
                }
            )

            # Create W&B Table for interactive prediction visualization (sample 500 points)
            sample_size = min(500, len(y_test))
            sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

            predictions_table = wandb.Table(
                columns=["true_rul", "predicted_rul", "error", "abs_error", "pct_error"],
                data=[
                    [
                        float(y_test_eval[i]),
                        float(y_pred[i]),
                        float(y_pred[i] - y_test_eval[i]),
                        float(abs(y_pred[i] - y_test_eval[i])),
                        float(abs(y_pred[i] - y_test_eval[i]) / max(y_test_eval[i], 1e-8) * 100),
                    ]
                    for i in sample_indices
                ],
            )

            wandb.log({"predictions_table": predictions_table})

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

    # Enhanced W&B run summary with comprehensive metadata
    summary_data = {
        # Model Architecture
        "model/architecture": model_name,
        "model/num_parameters": int(model.count_params()),
        "model/num_layers": len(model.layers),
        # Dataset Info
        "data/rul_min": float(y_min),
        "data/rul_max": float(y_max),
        "data/rul_range": float(y_max - y_min),
        "data/train_samples": int(len(y_train)),
        "data/input_timesteps": int(input_shape[0]),
        "data/num_features": int(input_shape[1]),
        # Training Config
        "training/epochs_trained": int(len(history.history["loss"])),
        "training/final_train_loss": float(history.history["loss"][-1]),
        "training/final_train_mae": float(history.history["mae"][-1]),
        "training/batch_size": int(config["batch_size"]),
        "training/learning_rate": float(config["learning_rate"]),
        "training/loss_name": config["loss_name"],
        "training/loss_alpha": float(config["loss_alpha"]),
        "training/loss_delta": float(config["loss_delta"]),
        "training/optimizer_name": config["optimizer_name"],
        "training/weight_decay": config["weight_decay"],
        "training/target_scaling": config["target_scaling"],
        "training/rul_clip_value": config["rul_clip_value"],
        "training/sample_weighting": config["sample_weighting"],
        "training/prediction_clip": config["prediction_clip"],
        "training/calibration_method": config["calibration_method"],
        "training/calibration": prediction_calibrator,
        # Reproducibility
        "reproducibility/seed": seed,
        "reproducibility/git_hash": git_hash,
    }

    # Add validation metrics if available
    if validation_data:
        summary_data.update(
            {
                "data/val_samples": int(len(y_val)),
                "training/final_val_loss": float(history.history["val_loss"][-1]),
                "training/final_val_mae": float(history.history["val_mae"][-1]),
            }
        )

    # Add test metrics if available
    if test_metrics:
        summary_data.update(
            {
                "data/test_samples": int(len(y_test)),
                "results/best_rmse": float(test_metrics["rmse"]),
                "results/best_mae": float(test_metrics["mae"]),
                "results/best_r2": float(test_metrics["r2"]),
                "results/best_phm_score": float(test_metrics["phm_score_normalized"]),
            }
        )

        add_sota_summary_metrics(summary_data, test_metrics, config)

    wandb.summary.update(summary_data)

    # Save model checkpoint with all artifacts
    if save_checkpoint and test_metrics:  # Only save if enabled and we have test results
        checkpoint_dir = save_model_checkpoint(
            model=model,
            scaler=scaler,
            config=config,
            test_metrics=test_metrics,
            model_name=model_name,
            run_name=run_name,
            y_min=y_min,
            y_max=y_max,
            target_transform=target_transform,
            prediction_calibrator=prediction_calibrator,
        )

        # Log checkpoint location to wandb
        wandb.log({"checkpoint/saved_path": checkpoint_dir})
        wandb.summary.update({"checkpoint/directory": checkpoint_dir})
    elif not save_checkpoint:
        print("\n⚠️  Model checkpoint saving disabled (--no-save flag)")

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
    save_checkpoint: bool = True,
    seed: int = 42,
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
                save_checkpoint=save_checkpoint,
                seed=seed,
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
        "Attention-based Models": [
            "transformer",
            "mdfa",
            "mdfa_paper",
            "cnn_lstm_attention",
            "cata_tcn",
            "ttsnet",
            "atcn",
            "sparse_transformer_bigrcu",
            "mstcn",
        ],
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
        "--loss",
        type=str,
        default="asymmetric_mse",
        choices=["asymmetric_mse", "asymmetric_huber", "mse", "mae", "huber", "log_cosh"],
        help="Training loss to optimize (default: asymmetric_mse)",
    )
    parser.add_argument(
        "--loss-alpha",
        type=float,
        default=2.0,
        help="Late-prediction penalty multiplier for asymmetric_mse (default: 2.0)",
    )
    parser.add_argument(
        "--loss-delta",
        type=float,
        default=1.0,
        help="Huber/asymmetric_huber delta in the active target scale (default: 1.0)",
    )
    parser.add_argument(
        "--rul-clip",
        type=float,
        default=None,
        help="Cap RUL labels at this value before training/evaluation (default: disabled)",
    )
    parser.add_argument(
        "--target-scaling",
        type=str,
        default="none",
        choices=["none", "minmax"],
        help="Optional target scaling strategy for training (default: none)",
    )
    parser.add_argument(
        "--gradient-clipnorm",
        type=float,
        default=None,
        help="Optional Adam clipnorm for training stability",
    )
    parser.add_argument(
        "--gradient-clipvalue",
        type=float,
        default=None,
        help="Optional Adam clipvalue for training stability",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optional AdamW weight decay",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="auto",
        help="Metric monitored by early stopping/LR scheduler, e.g. val_rmse or val_accuracy_10",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        default="auto",
        choices=["auto", "min", "max"],
        help="Monitor direction for callbacks (default: auto)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum monitored-metric improvement for early stopping",
    )
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
        "--lr-reduce-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate for ReduceLROnPlateau (default: 1e-7)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: None)",
    )
    parser.add_argument(
        "--sample-weighting",
        type=str,
        default="none",
        choices=["none", "low_rul", "inverse_rul", "balanced_bins"],
        help="Optional sample weighting strategy to emphasize hard/critical RUL regions",
    )
    parser.add_argument(
        "--sample-weight-strength",
        type=float,
        default=1.0,
        help="Strength for low_rul/inverse_rul sample weighting",
    )
    parser.add_argument(
        "--prediction-clip",
        type=str,
        default="none",
        choices=["none", "nonnegative", "train_range"],
        help="Clip predictions before metrics and saved inference metadata",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="none",
        choices=["none", "bias", "linear"],
        help="Validation-fitted prediction calibration applied before final metrics",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="JSON object of architecture-specific keyword args, e.g. '{\"dilation_rates\":[1,2,4,8]}'",
    )
    parser.add_argument(
        "--accuracy-recipe",
        type=str,
        default="none",
        choices=["none", "best"],
        help="Apply a bundled set of accuracy-focused training settings",
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
        "--visualize-attention",
        action="store_true",
        help="Generate attention/saliency interpretability report after training",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping to run all epochs",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable automatic model checkpoint saving",
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

    # Reproducibility options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Run N experiments with consecutive seeds [seed, seed+1, ..., seed+N-1] "
            "and report mean ± std (e.g. --seed 42 --num-seeds 5)"
        ),
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

    if args.accuracy_recipe != "none":
        recipe_config = get_accuracy_recipe_config(args.accuracy_recipe)
        json_training_config = {**recipe_config, **json_training_config}
        if not args.model and not args.compare and not args.compare_all:
            args.model = "mstcn"
        print(f"\nApplied accuracy recipe: {args.accuracy_recipe}")
        print(
            "Recipe strategies: short window, MSTCN scale kwargs, asymmetric Huber, "
            "AdamW, gradient clipping, validation RUL metrics, sample weighting, "
            "prediction clipping, validation calibration, and longer patience."
        )

    cli_model_kwargs = {}
    if args.model_kwargs:
        try:
            cli_model_kwargs = json.loads(args.model_kwargs)
            if not isinstance(cli_model_kwargs, dict):
                raise ValueError("--model-kwargs must decode to a JSON object")
        except Exception as e:
            print(f"Error parsing --model-kwargs: {e}")
            sys.exit(1)

    # Validate arguments
    if not args.compare and not args.compare_all and not args.model:
        parser.print_help()
        print("\n\nError: Must specify --model, --config, --compare, or --compare-all")
        sys.exit(1)

    # Load data
    print(f"\nLoading N-CMAPSS FD{args.fd} dataset...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=args.fd)
    val_X, val_y = val_pair if val_pair else (None, None)

    merged_model_kwargs = {
        **json_training_config.get("model_kwargs", {}),
        **cli_model_kwargs,
    }

    # Training configuration - JSON config overrides CLI args
    config = {
        "units": json_training_config.get("units", args.units),
        "dense_units": json_training_config.get("dense_units", args.dense_units),
        "dropout_rate": json_training_config.get("dropout_rate", args.dropout),
        "learning_rate": json_training_config.get("learning_rate", args.lr),
        "batch_size": json_training_config.get("batch_size", args.batch_size),
        "epochs": json_training_config.get("epochs", args.epochs),
        "loss_name": json_training_config.get(
            "loss_name", json_training_config.get("loss", args.loss)
        ),
        "loss_alpha": json_training_config.get("loss_alpha", args.loss_alpha),
        "loss_delta": json_training_config.get("loss_delta", args.loss_delta),
        "rul_clip_value": json_training_config.get("rul_clip_value", args.rul_clip),
        "target_scaling": json_training_config.get("target_scaling", args.target_scaling),
        "gradient_clipnorm": json_training_config.get("gradient_clipnorm", args.gradient_clipnorm),
        "gradient_clipvalue": json_training_config.get(
            "gradient_clipvalue", args.gradient_clipvalue
        ),
        "optimizer_name": json_training_config.get("optimizer_name", args.optimizer),
        "weight_decay": json_training_config.get("weight_decay", args.weight_decay),
        "max_sequence_length": json_training_config.get("max_sequence_length", args.max_seq_length),
        "patience_early_stop": json_training_config.get(
            "patience_early_stop", args.patience_early_stop
        ),
        "patience_lr_reduce": json_training_config.get(
            "patience_lr_reduce", args.patience_lr_reduce
        ),
        "lr_reduce_factor": json_training_config.get("lr_reduce_factor", args.lr_reduce_factor),
        "min_lr": json_training_config.get("min_lr", args.min_lr),
        "monitor_metric": json_training_config.get("monitor_metric", args.monitor_metric),
        "monitor_mode": json_training_config.get("monitor_mode", args.monitor_mode),
        "early_stop_min_delta": json_training_config.get(
            "early_stop_min_delta", args.early_stop_min_delta
        ),
        "sample_weighting": json_training_config.get("sample_weighting", args.sample_weighting),
        "sample_weight_strength": json_training_config.get(
            "sample_weight_strength", args.sample_weight_strength
        ),
        "sample_weight_power": json_training_config.get("sample_weight_power", 1.0),
        "sample_weight_min": json_training_config.get("sample_weight_min", 0.25),
        "sample_weight_max": json_training_config.get("sample_weight_max", 4.0),
        "prediction_clip": json_training_config.get("prediction_clip", args.prediction_clip),
        "calibration_method": json_training_config.get(
            "calibration_method", args.calibration_method
        ),
        "model_kwargs": merged_model_kwargs,
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
            save_checkpoint=not args.no_save,
            seed=args.seed,
        )

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: results/comparison/")
        print(f"Check wandb project: {args.project}-comparison")

    # Single model mode
    else:
        model_name = args.model
        seeds = (
            list(range(args.seed, args.seed + args.num_seeds))
            if args.num_seeds and args.num_seeds > 1
            else [args.seed]
        )

        if len(seeds) > 1:
            print(f"\nMulti-seed experiment: {len(seeds)} seeds {seeds}")

        all_metrics: List[Dict[str, float]] = []

        for seed in seeds:
            seed_suffix = f"-seed{seed}" if len(seeds) > 1 else ""
            base_run_name = args.run_name or f"{model_name}-run"
            run_name = f"{base_run_name}{seed_suffix}"

            print(f"\n{'='*80}")
            print(f"Training: {model_name}  (seed={seed})")
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
                run_name=run_name,
                normalize=not args.no_normalize,
                visualize=not args.no_visualize and len(seeds) == 1,
                use_early_stop=config.get("use_early_stop", True),
                save_checkpoint=not args.no_save,
                seed=seed,
            )

            if metrics:
                all_metrics.append(metrics)

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE: {model_name}")
        print("=" * 80)

        if all_metrics:
            if len(all_metrics) == 1:
                print("\nTest Set Metrics:")
                print(format_metrics(all_metrics[0]))
            else:
                # Multi-seed: report mean ± std
                print(f"\nMulti-Seed Results ({len(all_metrics)} seeds):")
                print("-" * 60)
                metric_keys = [
                    k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], float)
                ]
                for key in metric_keys:
                    values = [m[key] for m in all_metrics if key in m]
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values))
                    print(f"  {key:30s}: {mean_val:.4f} ± {std_val:.4f}")
                print("-" * 60)

                # Save multi-seed summary as JSON
                summary = make_json_safe(
                    {
                        "model": model_name,
                        "seeds": seeds,
                        "num_seeds": len(seeds),
                        "metrics_mean": {
                            k: float(np.mean([m[k] for m in all_metrics if k in m]))
                            for k in metric_keys
                        },
                        "metrics_std": {
                            k: float(np.std([m[k] for m in all_metrics if k in m]))
                            for k in metric_keys
                        },
                        "per_seed": {
                            str(seeds[i]): all_metrics[i] for i in range(len(all_metrics))
                        },
                    }
                )
                summary_path = f"results/{model_name}_multiseed_summary.json"
                os.makedirs("results", exist_ok=True)
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"\nMulti-seed summary saved to: {summary_path}")

        base_run_name = args.run_name or f"{model_name}-run"
        print(f"\nResults saved to: results/{base_run_name}/")
        print(f"Check wandb project: {args.project}")

        # Attention / interpretability report
        if args.visualize_attention and test_X is not None and all_metrics:
            from src.utils.interpretability import generate_interpretability_report

            print("\nGenerating interpretability report...")
            X_te, y_te = prepare_sequences(test_X, test_y, config.get("max_sequence_length"))
            if not args.no_normalize:
                X_tr, _ = prepare_sequences(dev_X, dev_y, config.get("max_sequence_length"))
                _, _, X_te, _ = normalize_data(X_tr, None, X_te)
            generate_interpretability_report(
                model=model,
                X_test=X_te,
                y_test=y_te,
                model_name=model_name,
                save_dir=f"results/{run_name}/attention",
            )


if __name__ == "__main__":
    main()
