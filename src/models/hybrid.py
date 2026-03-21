"""Hybrid CNN/RNN architectures."""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel
from src.models.registry import ModelRegistry


@ModelRegistry.register("cnn_lstm")
class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid."""

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
                layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(dropout_rate),
                layers.LSTM(units, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("cnn_gru")
class CNNGRUModel(BaseModel):
    """CNN-GRU hybrid."""

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
                layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(dropout_rate),
                layers.GRU(units, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("inception_lstm")
class InceptionLSTMModel(BaseModel):
    """Multi-branch convolutional frontend followed by LSTMs."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)
        conv1 = layers.Conv1D(filters=16, kernel_size=1, activation="relu", padding="same")(inputs)
        conv3 = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)
        conv5 = layers.Conv1D(filters=16, kernel_size=5, activation="relu", padding="same")(inputs)
        pool = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(inputs)
        pool = layers.Conv1D(filters=16, kernel_size=1, activation="relu", padding="same")(pool)

        x = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units // 2, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)
