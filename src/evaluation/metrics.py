"""
Evaluation metrics for RUL prediction.

Includes standard regression metrics and RUL-specific scoring functions.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PHM Society RUL Scoring Function.

    This is the official scoring function from the PHM08 Challenge.
    It penalizes late predictions (predicting failure after actual) more heavily
    than early predictions, as late predictions can lead to catastrophic failures.

    Score = sum(s_i) where:
        s_i = exp(-d/13) - 1  if d < 0 (early prediction)
        s_i = exp(d/10) - 1   if d >= 0 (late prediction)

    where d = y_pred - y_true (error)

    Lower score is better. A perfect prediction has score = 0.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(scores)


def phm_score_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized PHM Score (per sample average).

    Returns average PHM score per sample for easier interpretation.
    """
    return phm_score(y_true, y_pred) / len(y_true)


def asymmetric_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 2.0) -> float:
    """
    Asymmetric loss that penalizes late predictions more.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        alpha: Penalty multiplier for late predictions (default 2.0)
    """
    error = y_pred - y_true
    loss = np.where(error >= 0, alpha * error**2, error**2)
    return np.mean(loss)


def rul_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10.0
) -> float:
    """
    RUL Accuracy - percentage of predictions within threshold of true value.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        threshold: Acceptable error threshold (default 10 cycles)

    Returns:
        Percentage of predictions within threshold
    """
    abs_error = np.abs(y_true - y_pred)
    return (abs_error <= threshold).mean() * 100


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all RUL evaluation metrics.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values

    Returns:
        Dictionary with all metrics
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "phm_score_normalized": phm_score_normalized(y_true, y_pred),
        "asymmetric_loss": asymmetric_loss(y_true, y_pred),
        "accuracy_10": rul_accuracy(y_true, y_pred, threshold=10),
        "accuracy_15": rul_accuracy(y_true, y_pred, threshold=15),
        "accuracy_20": rul_accuracy(y_true, y_pred, threshold=20),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as a readable string."""
    lines = [
        "Evaluation Metrics:",
        "-" * 40,
        f"  MSE:              {metrics['mse']:.4f}",
        f"  RMSE:             {metrics['rmse']:.4f}",
        f"  MAE:              {metrics['mae']:.4f}",
        f"  MAPE:             {metrics['mape']:.2f}%",
        f"  R2 Score:         {metrics['r2']:.4f}",
        "-" * 40,
        f"  PHM Score:        {metrics['phm_score']:.2f}",
        f"  PHM Score (norm): {metrics['phm_score_normalized']:.4f}",
        f"  Asymmetric Loss:  {metrics['asymmetric_loss']:.4f}",
        "-" * 40,
        f"  Accuracy@10:      {metrics['accuracy_10']:.2f}%",
        f"  Accuracy@15:      {metrics['accuracy_15']:.2f}%",
        f"  Accuracy@20:      {metrics['accuracy_20']:.2f}%",
    ]
    return "\n".join(lines)


def compare_models(
    results: Dict[str, Dict[str, float]], primary_metric: str = "rmse"
) -> Tuple[str, Dict[str, float]]:
    """
    Compare multiple model results and find the best one.

    Args:
        results: Dictionary mapping model name to metrics dict
        primary_metric: Metric to use for comparison (lower is better for most metrics)

    Returns:
        Tuple of (best_model_name, best_model_metrics)
    """
    # Metrics where higher is better
    higher_is_better = {"r2", "accuracy_10", "accuracy_15", "accuracy_20"}

    best_model = None
    best_value = None

    for model_name, metrics in results.items():
        value = metrics.get(primary_metric)
        if value is None:
            continue

        if best_value is None:
            best_model = model_name
            best_value = value
        elif primary_metric in higher_is_better:
            if value > best_value:
                best_model = model_name
                best_value = value
        else:
            if value < best_value:
                best_model = model_name
                best_value = value

    return best_model, results[best_model]
