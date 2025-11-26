"""
Training visualization utilities for model evaluation and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history (loss and metrics over epochs).

    Args:
        history: Training history dictionary with loss/metrics
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(
            epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2
        )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (MSE)", fontsize=12)
    axes[0].set_title(f"{model_name} - Training Loss", fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot MAE
    axes[1].plot(epochs, history["mae"], "b-", label="Training MAE", linewidth=2)
    if "val_mae" in history:
        axes[1].plot(
            epochs, history["val_mae"], "r-", label="Validation MAE", linewidth=2
        )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("MAE", fontsize=12)
    axes[1].set_title(
        f"{model_name} - Mean Absolute Error", fontsize=14, fontweight="bold"
    )
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    max_samples: int = 500,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot prediction vs actual RUL values.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        model_name: Name of the model for title
        max_samples: Maximum samples to display for clarity
        save_path: Optional path to save the figure
    """
    # Limit samples for visualization
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, c="steelblue")
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("True RUL", fontsize=12)
    axes[0].set_ylabel("Predicted RUL", fontsize=12)
    axes[0].set_title(
        f"{model_name} - Prediction vs Actual", fontsize=14, fontweight="bold"
    )
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.5, s=20, c="steelblue")
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("True RUL", fontsize=12)
    axes[1].set_ylabel("Residual (Pred - True)", fontsize=12)
    axes[1].set_title(f"{model_name} - Residual Plot", fontsize=14, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Predictions plot saved to: {save_path}")

    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distribution of prediction errors.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error distribution
    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero Error")
    axes[0].axvline(
        x=errors.mean(),
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {errors.mean():.2f}",
    )
    axes[0].set_xlabel("Prediction Error (Pred - True)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        f"{model_name} - Error Distribution", fontsize=14, fontweight="bold"
    )
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Absolute error distribution
    axes[1].hist(abs_errors, bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes[1].axvline(
        x=abs_errors.mean(),
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {abs_errors.mean():.2f}",
    )
    axes[1].axvline(
        x=np.median(abs_errors),
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(abs_errors):.2f}",
    )
    axes[1].set_xlabel("Absolute Error", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(
        f"{model_name} - Absolute Error Distribution", fontsize=14, fontweight="bold"
    )
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error distribution saved to: {save_path}")

    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare multiple models across different metrics.

    Args:
        results: Dictionary mapping model name to metrics dict
        metrics: List of metrics to compare (default: rmse, mae, phm_score_normalized)
        save_path: Optional path to save the figure
    """
    if metrics is None:
        metrics = ["rmse", "mae", "phm_score_normalized", "accuracy_15"]

    model_names = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for idx, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]

        bars = axes[idx].bar(model_names, values, color=colors, edgecolor="black")
        axes[idx].set_xlabel("Model", fontsize=11)
        axes[idx].set_ylabel(metric.upper(), fontsize=11)
        axes[idx].set_title(f"{metric.upper()} Comparison", fontsize=12, fontweight="bold")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].grid(alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison saved to: {save_path}")

    plt.show()


def plot_sample_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_samples: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot RUL predictions for a sequence of samples.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        model_name: Name of the model for title
        n_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    n_samples = min(n_samples, len(y_true))
    indices = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(indices, y_true[:n_samples], "b-", label="True RUL", linewidth=2, alpha=0.8)
    ax.plot(
        indices, y_pred[:n_samples], "r--", label="Predicted RUL", linewidth=2, alpha=0.8
    )
    ax.fill_between(
        indices,
        y_true[:n_samples],
        y_pred[:n_samples],
        alpha=0.2,
        color="gray",
        label="Error",
    )

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("RUL (cycles)", fontsize=12)
    ax.set_title(
        f"{model_name} - Sample Predictions", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample predictions saved to: {save_path}")

    plt.show()


def create_evaluation_report(
    model_name: str,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: Dict[str, List[float]] = None,
    save_dir: str = "results",
) -> None:
    """
    Create a comprehensive evaluation report with all visualizations.

    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        y_true: True RUL values
        y_pred: Predicted RUL values
        history: Training history (optional)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluation Report: {model_name}")
    print("=" * 60)

    # Print metrics
    from src.utils.metrics import format_metrics

    print(format_metrics(metrics))

    # Generate visualizations
    if history:
        plot_training_history(
            history, model_name, save_path=f"{save_dir}/{model_name}_history.png"
        )

    plot_predictions(
        y_true, y_pred, model_name, save_path=f"{save_dir}/{model_name}_predictions.png"
    )

    plot_error_distribution(
        y_true, y_pred, model_name, save_path=f"{save_dir}/{model_name}_errors.png"
    )

    plot_sample_predictions(
        y_true, y_pred, model_name, save_path=f"{save_dir}/{model_name}_samples.png"
    )

    print(f"\nVisualizations saved to: {save_dir}/")
