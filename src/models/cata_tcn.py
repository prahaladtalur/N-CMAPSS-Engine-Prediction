"""Channel-and-Temporal Attention TCN (CATA-TCN) components."""

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
    """Dilated residual TCN block with dropout."""

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
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
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
        x = self.conv1(inputs)
        x = self.drop1(x, training=training)
        x = self.conv2(x)
        x = self.drop2(x, training=training)
        residual = self.proj(inputs) if self.proj is not None else inputs
        return layers.add([x, residual])


class ChannelAttention1D(layers.Layer):
    """Squeeze-excitation channel attention."""

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.fc1 = None
        self.fc2 = None

    def build(self, input_shape: Tuple[int, ...]):
        channels = int(input_shape[-1])
        reduced = max(channels // self.reduction_ratio, 1)
        self.fc1 = layers.Dense(reduced, activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pooled = tf.reduce_mean(inputs, axis=1)
        weights = self.fc2(self.fc1(pooled))
        return inputs * tf.expand_dims(weights, axis=1)


class TemporalAttention1D(layers.Layer):
    """Temporal attention over timesteps."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding="same", activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        temporal_map = self.conv(tf.concat([avg_pool, max_pool], axis=-1))
        return inputs * temporal_map


def build_cata_tcn_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    kernel_size: int = 3,
    num_layers: int = 4,
) -> keras.Model:
    """Build CATA-TCN model."""
    inputs = layers.Input(shape=input_shape)

    x = inputs
    for i in range(num_layers):
        x = ResidualTCNBlock(
            filters=units,
            kernel_size=kernel_size,
            dilation_rate=2**i,
            dropout_rate=dropout_rate,
        )(x)

    x = ChannelAttention1D()(x)
    x = TemporalAttention1D()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cata_tcn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=["mae", "mape"],
    )
    return model
