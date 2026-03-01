"""TTSNet: Transformer + TCN + Self-Attention fusion model for RUL."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def asymmetric_mse(alpha: float = 2.0):
    """Penalize late predictions more than early ones."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

    return loss


class ResidualTCNBlock(layers.Layer):
    """Lightweight residual TCN block used inside TTSNet."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)
        self.proj = None

    def build(self, input_shape: Tuple[int, ...]):
        if input_shape[-1] != self.filters:
            self.proj = layers.Conv1D(self.filters, kernel_size=1, padding="same")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.drop1(self.conv1(inputs), training=training)
        x = self.drop2(self.conv2(x), training=training)
        residual = self.proj(inputs) if self.proj is not None else inputs
        return layers.add([x, residual])


def _transformer_branch(
    inputs: tf.Tensor,
    units: int,
    num_heads: int,
    num_layers: int,
    dropout_rate: float,
) -> tf.Tensor:
    x = layers.Dense(units)(inputs)
    for _ in range(num_layers):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(units // num_heads, 1))(
            x, x
        )
        attn = layers.Dropout(dropout_rate)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = layers.Dense(units * 2, activation="relu")(x)
        ff = layers.Dense(units)(ff)
        ff = layers.Dropout(dropout_rate)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return layers.GlobalAveragePooling1D()(x)


def _tcn_branch(inputs: tf.Tensor, units: int, kernel_size: int, dropout_rate: float) -> tf.Tensor:
    x = inputs
    for i in range(3):
        x = ResidualTCNBlock(
            filters=units,
            kernel_size=kernel_size,
            dilation_rate=2**i,
            dropout_rate=dropout_rate,
        )(x)
    return layers.GlobalAveragePooling1D()(x)


def _self_attention_branch(
    inputs: tf.Tensor, units: int, num_heads: int, dropout_rate: float
) -> tf.Tensor:
    x = layers.Bidirectional(layers.GRU(units // 2, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(units // num_heads, 1))(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    return layers.GlobalMaxPooling1D()(x)


def build_ttsnet_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    kernel_size: int = 3,
) -> keras.Model:
    """Build TTSNet with three parallel branches and late fusion."""
    inputs = layers.Input(shape=input_shape)

    trans_feat = _transformer_branch(inputs, units, num_heads, num_transformer_layers, dropout_rate)
    tcn_feat = _tcn_branch(inputs, units, kernel_size, dropout_rate)
    sa_feat = _self_attention_branch(inputs, units, num_heads, dropout_rate)

    x = layers.concatenate([trans_feat, tcn_feat, sa_feat])
    x = layers.Dense(dense_units * 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ttsnet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=["mae", "mape"],
    )
    return model
