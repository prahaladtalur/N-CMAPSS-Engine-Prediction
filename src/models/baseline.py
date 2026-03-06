"""
Baseline models for RUL prediction.

Simple non-temporal models used as baselines for comparison.
"""

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("mlp")
class MLPModel(BaseModel):
    """
    Simple Multi-Layer Perceptron baseline.

    Flattens time series and uses only dense layers.
    Useful as a baseline to compare against temporal models.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_hidden_layers: int = 3,
    ) -> keras.Model:
        model = keras.Sequential([layers.Input(shape=input_shape)])

        # Flatten the time series
        model.add(layers.Flatten())

        # Hidden layers
        for i in range(num_hidden_layers):
            layer_units = units // (2**i) if i > 0 else units
            model.add(layers.Dense(layer_units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))

        # Final dense layer
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation="linear"))

        return BaseModel.compile_model(model, learning_rate)
