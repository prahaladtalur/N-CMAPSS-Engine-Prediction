"""
Attention-Based Temporal Convolutional Network (ATCN) for RUL prediction.

Paper: An attention-based temporal convolutional network method for predicting
       remaining useful life of aero-engine (2023)

Architecture:
    Input → Improved Self-Attention → TCN Blocks (dilated residual)
          → Channel Attention (squeeze-excitation) → Dense → RUL

Key components:
    - Improved Self-Attention (ISA): Weights timestep contributions with
      learnable position embeddings and layer normalization
    - TCN Backbone: Dilated causal convolutions in residual blocks
    - Squeeze-and-Excitation: Channel attention to weight feature importance
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.cata_tcn import ResidualTCNBlock, ChannelAttention1D


def asymmetric_mse(alpha: float = 2.0):
    """Penalize late predictions more than early ones."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

    return loss


class ImprovedSelfAttention(layers.Layer):
    """
    Improved self-attention mechanism that weights timestep contributions.

    Differences from standard self-attention:
    - Uses learnable position embeddings
    - Applies layer normalization before attention
    - Includes residual connection for stability
    - Multi-head attention with scaled dot-product
    """

    def __init__(self, units: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads, dropout=0.1
        )
        self.add = layers.Add()
        self.position_embedding = None

    def build(self, input_shape: Tuple[int, ...]):
        seq_length = input_shape[1]
        feature_dim = input_shape[2]

        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(seq_length, feature_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x_with_pos = inputs + self.position_embedding

        x_norm = self.layer_norm(x_with_pos)

        attn_output = self.mha(
            query=x_norm, value=x_norm, key=x_norm, training=training, return_attention_scores=False
        )

        output = self.add([inputs, attn_output])

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "num_heads": self.num_heads})
        return config


def build_atcn_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    num_heads: int = 4,
    kernel_size: int = 3,
    num_tcn_layers: int = 4,
) -> keras.Model:
    """
    Build ATCN model for RUL prediction.

    Args:
        input_shape: (sequence_length, num_features)
        units: Number of filters in TCN blocks
        dense_units: Number of units in final dense layer
        dropout_rate: Dropout probability
        learning_rate: Optimizer learning rate
        num_heads: Number of attention heads in ISA
        kernel_size: Kernel size for TCN convolutions
        num_tcn_layers: Number of stacked TCN blocks

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name="input")

    x = ImprovedSelfAttention(units=units, num_heads=num_heads, name="improved_self_attention")(
        inputs
    )

    for i in range(num_tcn_layers):
        x = ResidualTCNBlock(
            filters=units,
            kernel_size=kernel_size,
            dilation_rate=2**i,
            dropout_rate=dropout_rate,
            name=f"tcn_block_{i}",
        )(x)

    x = ChannelAttention1D(reduction_ratio=16, name="channel_attention")(x)

    x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="atcn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=["mae", "mape"],
    )
    return model
