"""
Recurrent Neural Network (RNN) architectures for RUL prediction.

Includes:
- LSTM: Standard Long Short-Term Memory
- BiLSTM: Bidirectional LSTM
- GRU: Gated Recurrent Unit
- BiGRU: Bidirectional GRU
"""

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("lstm")
class LSTMModel(BaseModel):
    """Standard LSTM model for RUL prediction."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(units, return_sequences=True),
                layers.Dropout(dropout_rate),
                layers.LSTM(units // 2, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("bilstm")
class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model - captures both past and future context."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
                layers.Dropout(dropout_rate),
                layers.Bidirectional(layers.LSTM(units // 2, return_sequences=False)),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("gru")
class GRUModel(BaseModel):
    """GRU model - simpler than LSTM, often faster with similar performance."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.GRU(units, return_sequences=True),
                layers.Dropout(dropout_rate),
                layers.GRU(units // 2, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("bigru")
class BiGRUModel(BaseModel):
    """Bidirectional GRU model."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Bidirectional(layers.GRU(units, return_sequences=True)),
                layers.Dropout(dropout_rate),
                layers.Bidirectional(layers.GRU(units // 2, return_sequences=False)),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)
