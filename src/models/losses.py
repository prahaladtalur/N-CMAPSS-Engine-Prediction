"""
Custom loss functions for RUL prediction.

Standard MSE treats all errors equally, but for RUL prediction:
1. Errors in critical zones (low RUL) are more costly
2. Under-prediction (predicting too high RUL) is more dangerous than over-prediction
3. Asymmetric penalty based on PHM competition scoring

These loss functions emphasize accuracy where it matters most.
"""

import tensorflow as tf
from tensorflow import keras


def phm_score_loss(y_true, y_pred):
    """
    PHM (Prognostics and Health Management) scoring function as loss.

    Asymmetric penalty that heavily penalizes late predictions:
    - Under-prediction (pred > true): e^(error/10) - 1  (late failure prediction = BAD)
    - Over-prediction (pred < true): e^(-error/13) - 1 (early prediction = less bad)

    This is based on the PHM08 competition scoring function.

    Returns:
        Lower is better. Heavily penalizes predicting failure too late.
    """
    error = y_pred - y_true

    # Asymmetric penalty
    late_prediction = tf.exp(error / 10.0) - 1.0  # pred > true (bad!)
    early_prediction = tf.exp(-error / 13.0) - 1.0  # pred < true (ok)

    # Use late penalty when pred > true, early penalty otherwise
    penalty = tf.where(error > 0, late_prediction, early_prediction)

    return tf.reduce_mean(penalty)


def weighted_mse_loss(critical_threshold=30, critical_weight=3.0):
    """
    MSE with higher weight for critical zone (low RUL values).

    Args:
        critical_threshold: RUL threshold below which errors are weighted more (default: 30)
        critical_weight: Multiplier for critical zone errors (default: 3.0)

    Returns:
        Loss function that emphasizes critical zone accuracy.
    """
    def loss(y_true, y_pred):
        # Standard squared error
        squared_error = tf.square(y_pred - y_true)

        # Apply higher weight to critical zone
        is_critical = tf.cast(y_true < critical_threshold, tf.float32)
        weights = is_critical * critical_weight + (1 - is_critical) * 1.0

        weighted_error = squared_error * weights
        return tf.reduce_mean(weighted_error)

    return loss


def asymmetric_mse_loss(under_prediction_weight=2.0):
    """
    Asymmetric MSE that penalizes under-prediction more than over-prediction.

    Under-prediction = predicting RUL too high = predicting failure too late = DANGEROUS
    Over-prediction = predicting RUL too low = predicting failure too early = SAFE

    Args:
        under_prediction_weight: Multiplier for under-prediction errors (default: 2.0)

    Returns:
        Loss function with asymmetric penalty.
    """
    def loss(y_true, y_pred):
        error = y_pred - y_true
        squared_error = tf.square(error)

        # Penalize under-prediction (pred > true) more
        is_under_predicted = tf.cast(error > 0, tf.float32)
        weights = is_under_predicted * under_prediction_weight + (1 - is_under_predicted) * 1.0

        weighted_error = squared_error * weights
        return tf.reduce_mean(weighted_error)

    return loss


def combined_rul_loss(
    critical_threshold=30,
    critical_weight=3.0,
    under_prediction_weight=2.0,
    alpha=0.7,
):
    """
    Combined loss: weighted MSE + asymmetric penalty.

    This combines:
    1. Critical zone weighting (errors at low RUL matter more)
    2. Asymmetric penalty (under-prediction is worse than over-prediction)

    Args:
        critical_threshold: RUL threshold for critical zone (default: 30)
        critical_weight: Weight for critical zone errors (default: 3.0)
        under_prediction_weight: Weight for under-predictions (default: 2.0)
        alpha: Balance between weighted and asymmetric loss (default: 0.7)

    Returns:
        Combined loss function emphasizing critical accuracy and safe predictions.
    """
    def loss(y_true, y_pred):
        error = y_pred - y_true
        squared_error = tf.square(error)

        # Critical zone weighting
        is_critical = tf.cast(y_true < critical_threshold, tf.float32)
        critical_weights = is_critical * critical_weight + (1 - is_critical) * 1.0

        # Asymmetric weighting
        is_under_predicted = tf.cast(error > 0, tf.float32)
        asymmetric_weights = (
            is_under_predicted * under_prediction_weight + (1 - is_under_predicted) * 1.0
        )

        # Combine both weightings
        combined_weights = critical_weights * asymmetric_weights
        weighted_error = squared_error * combined_weights

        # Balance with standard MSE
        weighted_mse = tf.reduce_mean(weighted_error)
        standard_mse = tf.reduce_mean(squared_error)

        return alpha * weighted_mse + (1 - alpha) * standard_mse

    return loss


def quantile_loss(quantile=0.9):
    """
    Quantile loss for uncertainty-aware predictions.

    Useful for conservative RUL estimates (e.g., 90th percentile = safe estimate).

    Args:
        quantile: Desired quantile (default: 0.9 for conservative estimates)

    Returns:
        Quantile loss function.
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred

        # Asymmetric penalty based on quantile
        positive_error = quantile * error
        negative_error = (quantile - 1) * error

        quantile_error = tf.where(error >= 0, positive_error, negative_error)
        return tf.reduce_mean(quantile_error)

    return loss


# Registry of loss functions
LOSS_FUNCTIONS = {
    "mse": "mse",  # Standard MSE (baseline)
    "phm_score": phm_score_loss,  # PHM competition scoring
    "weighted_mse": weighted_mse_loss(),  # Critical zone emphasis
    "asymmetric_mse": asymmetric_mse_loss(),  # Under-prediction penalty
    "combined_rul": combined_rul_loss(),  # Combined (RECOMMENDED)
    "quantile_90": quantile_loss(0.9),  # Conservative estimates
}


def get_loss_function(name="combined_rul", **kwargs):
    """
    Get loss function by name.

    Args:
        name: Loss function name
        **kwargs: Arguments for parameterized loss functions

    Returns:
        Loss function
    """
    if name in ["mse", "phm_score"]:
        return LOSS_FUNCTIONS[name]
    elif name == "weighted_mse":
        return weighted_mse_loss(**kwargs)
    elif name == "asymmetric_mse":
        return asymmetric_mse_loss(**kwargs)
    elif name == "combined_rul":
        return combined_rul_loss(**kwargs)
    elif name.startswith("quantile"):
        q = float(name.split("_")[1]) if "_" in name else 0.9
        return quantile_loss(q)
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_loss_recommendations():
    """Get recommendations for different use cases."""
    return {
        "best_overall": "combined_rul",
        "maximum_safety": "asymmetric_mse",
        "critical_zone_focus": "weighted_mse",
        "phm_competition": "phm_score",
        "conservative_estimates": "quantile_90",
        "baseline": "mse",
    }
