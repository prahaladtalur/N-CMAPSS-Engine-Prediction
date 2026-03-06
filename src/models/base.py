"""
Base model class and common utilities for RUL prediction models.

All model architectures inherit from BaseModel and use the asymmetric_mse loss
which penalizes late predictions more heavily than early predictions.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import tensorflow as tf
from tensorflow import keras


def asymmetric_mse(alpha: float = 2.0):
    """
    Asymmetric MSE loss that penalizes late RUL predictions more heavily.

    In RUL prediction, over-predicting remaining life (y_pred > y_true) is
    dangerous — it risks operating past failure. This loss applies a penalty
    multiplier of ``alpha`` to squared errors when the prediction exceeds the
    true value, while standard squared error is used for early predictions.

    Args:
        alpha: Penalty multiplier for late predictions (default 2.0).
            - alpha=1.0: Standard MSE (no asymmetry)
            - alpha=2.0: Late predictions penalized 2× (recommended)
            - alpha>2.0: Even more conservative (very safety-critical)

    Returns:
        Loss function compatible with Keras model.compile()

    Example:
        >>> model.compile(optimizer='adam', loss=asymmetric_mse(alpha=2.0))
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        # Penalize positive errors (late predictions) more heavily
        return tf.reduce_mean(
            tf.where(error >= 0, alpha * tf.square(error), tf.square(error))
        )

    return loss


class BaseModel(ABC):
    """
    Abstract base class for all RUL prediction models.

    All models must implement the build() method with a consistent signature
    and should use compile_model() for compilation.
    """

    @staticmethod
    @abstractmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> keras.Model:
        """
        Build and compile the model.

        Args:
            input_shape: (timesteps, features)
            units: Number of units in recurrent/temporal layers
            dense_units: Number of units in final dense layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            **kwargs: Additional model-specific parameters

        Returns:
            Compiled Keras model
        """
        pass

    @staticmethod
    def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
        """
        Compile model with standard settings for RUL prediction.

        Uses:
        - Optimizer: Adam with given learning rate
        - Loss: Asymmetric MSE (penalizes late predictions 2×)
        - Metrics: MAE, MAPE

        Args:
            model: Uncompiled Keras model
            learning_rate: Learning rate for Adam optimizer

        Returns:
            Compiled Keras model
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=asymmetric_mse(),
            metrics=["mae", "mape"],
        )
        return model
