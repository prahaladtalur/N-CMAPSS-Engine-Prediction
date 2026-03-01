"""
Sparse Transformer with Bidirectional Gated Recurrent Convolution Unit (Bi-GRCU).

Paper: A sequence ensemble method based on Sparse Transformer with bidirectional
       gated recurrent convolution unit for RUL prediction (2025)

Architecture:
    Input → Dual branches:
            1. Bi-GRCU branch (short-term dependencies)
            2. Sparse Transformer branch (long-term dependencies with LRLS-Attention)
          → Concatenate → Dense layers → RUL

Key innovations:
    - Bi-GRCU: Gated fusion of Bidirectional GRU and Conv1D features
    - LRLS-Attention: Long-Range Locality Sparse attention pattern
      (local window + global tokens) reduces complexity from O(T²) to O(T×(k+g))
    - Ensemble fusion combines short-term and long-term modeling
"""

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


class BiGRCU(layers.Layer):
    """
    Bidirectional Gated Recurrent Convolution Unit.

    Combines recurrent (Bi-GRU) and convolutional operations for short-term dependencies:
    - Bi-GRU captures bidirectional sequential patterns
    - Conv1D extracts local spatial features
    - Learnable gating mechanism fuses RNN and CNN features
    """

    def __init__(self, units: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size

        # Bidirectional GRU for sequential dependencies
        self.bigru = layers.Bidirectional(
            layers.GRU(units, return_sequences=True, name="gru"), name="bidirectional_gru"
        )

        # Conv1D for local pattern extraction
        self.conv = layers.Conv1D(
            filters=units * 2,  # Match Bi-GRU output size
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name="conv1d",
        )

        # Learnable gate for fusion
        self.gate = layers.Dense(units * 2, activation="sigmoid", name="fusion_gate")

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        # RNN branch: bidirectional sequential modeling
        rnn_out = self.bigru(inputs, training=training)

        # CNN branch: local convolutional features
        conv_out = self.conv(inputs, training=training)

        # Gated fusion: learn optimal combination
        gate_weights = self.gate(rnn_out)
        fused = gate_weights * rnn_out + (1 - gate_weights) * conv_out

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "kernel_size": self.kernel_size})
        return config


class LRLSAttention(layers.Layer):
    """
    Long-Range Locality Sparse (LRLS) Attention mechanism.

    Implements sparse attention pattern to reduce computational complexity:
    - Local attention: each token attends to k nearest neighbors
    - Global attention: each token attends to g global tokens (first g positions)
    - Complexity: O(T × (k + g)) instead of O(T²) for full attention

    This enables efficient processing of long sequences while maintaining
    both local and global context.
    """

    def __init__(
        self,
        num_heads: int = 4,
        local_window: int = 32,
        num_global_tokens: int = 8,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.local_window = local_window
        self.num_global_tokens = num_global_tokens
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=None,  # Will be set in build()
            dropout=dropout_rate,
            name="multi_head_attention",
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="layernorm1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="layernorm2")
        self.ffn = None
        self.dropout = layers.Dropout(dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[int, ...]):
        feature_dim = input_shape[-1]
        key_dim = feature_dim // self.num_heads

        # Override MHA with correct key_dim
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name="multi_head_attention",
        )

        # Feed-forward network
        self.ffn = keras.Sequential(
            [
                layers.Dense(feature_dim * 4, activation="relu", name="ffn_dense1"),
                layers.Dropout(self.dropout_rate, name="ffn_dropout"),
                layers.Dense(feature_dim, name="ffn_dense2"),
            ],
            name="feed_forward",
        )

        super().build(input_shape)

    def create_sparse_attention_mask(self, seq_length: int) -> tf.Tensor:
        """
        Create sparse attention mask for LRLS pattern.

        For each position i:
        - Can attend to positions [max(0, i-k/2), min(T, i+k/2)] (local window)
        - Can attend to first g positions (global tokens)

        Returns:
            attention_mask: (seq_length, seq_length) boolean mask
        """
        # Create full False mask (all positions masked)
        mask = tf.zeros((seq_length, seq_length), dtype=tf.bool)

        # Local window attention
        local_radius = self.local_window // 2
        for i in range(seq_length):
            start = max(0, i - local_radius)
            end = min(seq_length, i + local_radius + 1)

            # Mark local window as True (unmasked)
            indices = tf.range(start, end)
            updates = tf.ones(end - start, dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(
                mask, tf.stack([tf.fill([end - start], i), indices], axis=1), updates
            )

        # Global token attention (first g tokens attend to all, all attend to first g)
        if self.num_global_tokens > 0:
            g = min(self.num_global_tokens, seq_length)
            # First g tokens can attend to all positions
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.reshape(tf.range(g * seq_length), [g * seq_length, 1]),
                tf.ones(g * seq_length, dtype=tf.bool),
            )
            # All positions can attend to first g tokens
            for i in range(seq_length):
                indices = tf.range(g)
                updates = tf.ones(g, dtype=tf.bool)
                mask = tf.tensor_scatter_nd_update(
                    mask, tf.stack([tf.fill([g], i), indices], axis=1), updates
                )

        # Convert to attention mask format (1.0 for attended, 0.0 for masked)
        return tf.cast(mask, tf.float32)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        seq_length = tf.shape(inputs)[1]

        # Create sparse attention mask
        attention_mask = self.create_sparse_attention_mask(seq_length)
        attention_mask = tf.expand_dims(attention_mask, 0)  # Add batch dimension

        # Layer normalization
        x_norm = self.layernorm1(inputs)

        # Sparse multi-head attention with mask
        attn_output = self.mha(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            attention_mask=attention_mask,
            training=training,
        )

        # Residual connection
        x = inputs + self.dropout(attn_output, training=training)

        # Feed-forward network with residual
        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + self.dropout(ffn_output, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "local_window": self.local_window,
                "num_global_tokens": self.num_global_tokens,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def build_sparse_transformer_bigrcu_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    local_window: int = 32,
    num_global_tokens: int = 8,
) -> keras.Model:
    """
    Build Sparse Transformer + Bi-GRCU ensemble model for RUL prediction.

    Args:
        input_shape: (sequence_length, num_features)
        units: Number of units for Bi-GRCU and Transformer
        dense_units: Number of units in final dense layers
        dropout_rate: Dropout probability
        learning_rate: Optimizer learning rate
        num_heads: Number of attention heads in Sparse Transformer
        num_transformer_layers: Number of stacked Sparse Transformer layers
        local_window: Size of local attention window (k)
        num_global_tokens: Number of global tokens (g)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Branch 1: Bi-GRCU for short-term dependencies
    bigrcu_out = BiGRCU(units=units, kernel_size=3, name="bigrcu")(inputs)
    bigrcu_features = layers.GlobalAveragePooling1D(name="bigrcu_pooling")(bigrcu_out)

    # Branch 2: Sparse Transformer for long-term dependencies
    x = inputs
    for i in range(num_transformer_layers):
        x = LRLSAttention(
            num_heads=num_heads,
            local_window=local_window,
            num_global_tokens=num_global_tokens,
            dropout_rate=dropout_rate,
            name=f"lrls_attention_{i}",
        )(x)

    sparse_features = layers.GlobalAveragePooling1D(name="sparse_pooling")(x)

    # Ensemble fusion: concatenate both branches
    fused = layers.concatenate([bigrcu_features, sparse_features], name="ensemble_fusion")

    # Dense output layers
    x = layers.Dense(dense_units * 2, activation="relu", name="dense_1")(fused)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sparse_transformer_bigrcu")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=["mae", "mape"],
    )
    return model
