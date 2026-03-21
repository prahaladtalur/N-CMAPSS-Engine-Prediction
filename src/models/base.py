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


def get_loss_function(loss_name: str = "asymmetric_mse", loss_alpha: float = 2.0):
    """Resolve a configured training loss."""
    losses = {
        "asymmetric_mse": asymmetric_mse(alpha=loss_alpha),
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
