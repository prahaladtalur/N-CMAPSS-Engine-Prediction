"""Hybrid CNN/RNN architectures."""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel, compile_model_for_training
from src.models.registry import ModelRegistry


class _CBAMBlock(layers.Layer):
    """Convolutional Block Attention Module (CBAM).

    Applies sequential channel attention then temporal attention to a feature map.
    Based on: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018),
    adapted for 1-D time-series as in Li et al. (PeerJ CS 2022) which achieved
    RMSE 5.50 on N-CMAPSS DS02.

    Input shape:  (batch, timesteps, filters)
    Output shape: (batch, timesteps, filters)  — same, attention-refined
    """

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        bottleneck = max(1, filters // self.reduction_ratio)
        # Channel attention shared MLP
        self.ca_dense1 = layers.Dense(bottleneck, activation="relu", use_bias=False)
        self.ca_dense2 = layers.Dense(filters, use_bias=False)
        self.ca_sigmoid = layers.Activation("sigmoid")
        # Temporal attention conv
        self.ta_conv = layers.Conv1D(1, kernel_size=7, padding="same", activation="sigmoid")
        self.concat = layers.Concatenate(axis=-1)
        super().build(input_shape)

    def call(self, x):
        import keras
        # --- Channel attention ---
        avg_pool = keras.ops.mean(x, axis=1, keepdims=True)   # (B, 1, C)
        max_pool = keras.ops.max(x, axis=1, keepdims=True)    # (B, 1, C)
        channel_attn = self.ca_sigmoid(
            self.ca_dense2(self.ca_dense1(avg_pool))
            + self.ca_dense2(self.ca_dense1(max_pool))
        )
        x = x * channel_attn

        # --- Temporal attention ---
        avg_t = keras.ops.mean(x, axis=-1, keepdims=True)    # (B, T, 1)
        max_t = keras.ops.max(x, axis=-1, keepdims=True)     # (B, T, 1)
        temporal_attn = self.ta_conv(self.concat([avg_t, max_t]))
        x = x * temporal_attn
        return x

    def get_config(self):
        return {**super().get_config(), "reduction_ratio": self.reduction_ratio}


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


@ModelRegistry.register("cbam_cnn_lstm")
class CBAMCNNLSTMModel(BaseModel):
    """CNN-LSTM with Convolutional Block Attention Module (CBAM).

    Replicates the architecture from Li et al. (PeerJ CS 2022), which achieved
    RMSE 5.50 on N-CMAPSS DS02 — the best published absolute-cycle result at the
    time of its release.  CBAM adds two sequential attention passes after the CNN
    feature extractor:
      1. Channel attention  — which sensors matter most.
      2. Temporal attention — which timesteps matter most.
    Both passes are lightweight (< 1% parameter overhead) and can be applied to
    any CNN feature map without changing the downstream LSTM.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        cnn_filters: int = 64,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        # Multi-scale CNN feature extractor (3 kernel sizes as in the paper)
        x = layers.Conv1D(cnn_filters, kernel_size=7, padding="same", activation="relu")(inputs)
        x = layers.Conv1D(cnn_filters, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.Conv1D(cnn_filters, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # CBAM: channel attention then temporal attention
        x = _CBAMBlock(reduction_ratio=8)(x)

        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return compile_model_for_training(model, learning_rate=learning_rate)
