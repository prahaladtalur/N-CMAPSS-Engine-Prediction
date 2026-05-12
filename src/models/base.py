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


def asymmetric_huber(alpha: float = 2.0, delta: float = 1.0):
    """Huber loss with a larger penalty for late RUL predictions."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + delta * linear
        return tf.reduce_mean(tf.where(error >= 0, alpha * huber, huber))

    return loss


def get_loss_function(
    loss_name: str = "asymmetric_mse",
    loss_alpha: float = 2.0,
    loss_delta: float = 1.0,
):
    """Resolve a configured training loss."""
    losses = {
        "asymmetric_mse": asymmetric_mse(alpha=loss_alpha),
        "mse": keras.losses.MeanSquaredError(),
        "mae": keras.losses.MeanAbsoluteError(),
        "huber": keras.losses.Huber(delta=loss_delta),
        "asymmetric_huber": asymmetric_huber(alpha=loss_alpha, delta=loss_delta),
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
    loss_delta: float = 1.0,
    gradient_clipnorm: Optional[float] = None,
    gradient_clipvalue: Optional[float] = None,
    optimizer_name: str = "adam",
    weight_decay: Optional[float] = None,
) -> keras.Model:
    """Compile a built model with configurable optimizer and loss settings."""
    optimizer_kwargs = {"learning_rate": learning_rate}
    if gradient_clipnorm is not None:
        optimizer_kwargs["clipnorm"] = gradient_clipnorm
    if gradient_clipvalue is not None:
        optimizer_kwargs["clipvalue"] = gradient_clipvalue

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(**optimizer_kwargs)
    elif optimizer_name == "adamw":
        if weight_decay is None:
            weight_decay = 0.0
        optimizer = keras.optimizers.AdamW(weight_decay=weight_decay, **optimizer_kwargs)
    else:
        raise ValueError("Unsupported optimizer. Available: adam, adamw")

    model.compile(
        optimizer=optimizer,
        loss=get_loss_function(
            loss_name=loss_name,
            loss_alpha=loss_alpha,
            loss_delta=loss_delta,
        ),
        metrics=["mae", "mape"],
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
