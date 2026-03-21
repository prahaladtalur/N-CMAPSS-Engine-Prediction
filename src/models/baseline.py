"""Baseline non-temporal architectures."""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel
from src.models.registry import ModelRegistry


@ModelRegistry.register("mlp")
class MLPModel(BaseModel):
    """Simple MLP baseline."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_hidden_layers: int = 3,
    ) -> keras.Model:
        model = keras.Sequential([layers.Input(shape=input_shape), layers.Flatten()])
        for i in range(num_hidden_layers):
            layer_units = units // (2**i) if i > 0 else units
            model.add(layers.Dense(layer_units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation="linear"))
        return BaseModel.compile_model(model, learning_rate)
