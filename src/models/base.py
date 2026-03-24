"""Shared model base classes and training helpers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


def asymmetric_mse(alpha: float = 2.0):
    """Penalize late RUL predictions more heavily than early ones."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

    return loss


def multi_zone_mse(early_alpha: float = 2.0, near_alpha: float = 3.5, critical_alpha: float = 6.0,
                   near_threshold: float = 40.0, critical_threshold: float = 20.0):
    """Zone-aware asymmetric loss that penalises over-prediction more heavily as RUL approaches zero.

    Three zones (tuned for N-CMAPSS max_rul=65):
      - Early      (RUL > near_threshold):       penalty factor = early_alpha
      - Near-end   (critical < RUL ≤ near):      penalty factor = near_alpha
      - Critical   (RUL ≤ critical_threshold):   penalty factor = critical_alpha

    Over-prediction (y_pred > y_true, i.e. predicting too much life) uses the zone penalty.
    Under-prediction always uses squared error without extra weighting.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true  # positive = over-prediction (dangerous)

        # Zone membership based on true RUL
        in_critical = tf.cast(y_true <= critical_threshold, tf.float32)
        in_near = tf.cast(
            tf.logical_and(y_true > critical_threshold, y_true <= near_threshold), tf.float32
        )
        in_early = tf.cast(y_true > near_threshold, tf.float32)

        # Alpha per sample: blend zone weights
        alpha = in_early * early_alpha + in_near * near_alpha + in_critical * critical_alpha

        # Apply zone alpha only to over-predictions; under-predictions use plain squared error
        weighted = tf.where(error >= 0, alpha * tf.square(error), tf.square(error))
        return tf.reduce_mean(weighted)

    return loss


def get_loss_function(loss_name: str = "asymmetric_mse", loss_alpha: float = 2.0):
    """Resolve a configured training loss."""
    losses = {
        "asymmetric_mse": asymmetric_mse(alpha=loss_alpha),
        "multi_zone_mse": multi_zone_mse(),
        "mse": keras.losses.MeanSquaredError(),
        "mae": keras.losses.MeanAbsoluteError(),
        "huber": keras.losses.Huber(),
        "log_cosh": keras.losses.LogCosh(),
    }
    if loss_name not in losses:
        available = ", ".join(sorted(losses))
        raise ValueError(f"Unsupported loss '{loss_name}'. Available: {available}")
    return losses[loss_name]


def compile_model_for_training(
    model: keras.Model,
    learning_rate: float,
    loss_name: str = "asymmetric_mse",
    loss_alpha: float = 2.0,
    gradient_clipnorm: Optional[float] = None,
    gradient_clipvalue: Optional[float] = None,
) -> keras.Model:
    """Compile a built model with configurable optimizer and loss settings."""
    optimizer_kwargs = {"learning_rate": learning_rate}
    if gradient_clipnorm is not None:
        optimizer_kwargs["clipnorm"] = gradient_clipnorm
    if gradient_clipvalue is not None:
        optimizer_kwargs["clipvalue"] = gradient_clipvalue

    optimizer = keras.optimizers.Adam(**optimizer_kwargs)
    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(loss_name=loss_name, loss_alpha=loss_alpha),
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            "mae",
            "mape",
        ],
    )
    return model


class BaseModel(ABC):
    """Base class for all models."""

    @staticmethod
    @abstractmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        """Build and compile the model."""

    @staticmethod
    def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
        """Compile model with standard settings."""
        return compile_model_for_training(model, learning_rate=learning_rate)
