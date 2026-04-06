"""STAR: Two-Stage Attention-based Hierarchical Transformer for RUL prediction.

Based on: "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan
Engine Remaining Useful Life Prediction" (Sensors, MDPI 2024)
DOI: 10.3390/s24030824

Key insight: Standard transformers apply self-attention across the combined
time × sensor space, which mixes two fundamentally different dependency types.
STAR separates them into two dedicated stages per block:

  Stage 1 — Sensor-wise attention:
    Transpose (B,T,F) → (B,F,T); apply MHA treating each sensor as a token.
    Sensors can now attend to each other, capturing inter-sensor correlations
    (e.g. fan speed and compressor outlet temperature co-degrade).

  Stage 2 — Temporal attention:
    Standard self-attention on (B,T,F); timesteps attend to each other,
    capturing degradation dynamics over time.

Each stage has its own residual connection, LayerNorm, and feed-forward network.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel, compile_model_for_training
from src.models.registry import ModelRegistry


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class STARBlock(layers.Layer):
    """Two-stage attention block: sensor-wise MHA then temporal MHA.

    Stage 1: (B,T,F) → (B,F,T) → sensor MHA → (B,F,T) → (B,T,F)
    Stage 2: temporal MHA on (B,T,F)
    Each stage: Add & Norm → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else d_model * 2
        self.dropout_rate = dropout

        key_dim = max(d_model // num_heads, 1)

        # Stage 1: sensor-wise attention (across feature/sensor dimension)
        # Note: after transposing (B,T,F)→(B,F,T), each sensor token has dim T.
        # The sensor MHA uses key_dim for attention; output_shape=None keeps output
        # dimension matching input token dimension (T), so the residual add works.
        self.sensor_mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout, name="sensor_mha"
        )
        self.sensor_norm1 = layers.LayerNormalization(epsilon=1e-6)
        # sensor_ffn is built lazily in build() because output dim = timesteps (T), not d_model
        self.sensor_ffn = None
        self.sensor_norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Stage 2: temporal attention (across time dimension)
        self.temporal_mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout, name="temporal_mha"
        )
        self.temporal_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.temporal_ffn = keras.Sequential(
            [
                layers.Dense(self.ffn_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ],
            name="temporal_ffn",
        )
        self.temporal_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape: Tuple):
        # input_shape: (batch, timesteps, d_model)
        # After transpose (B,T,F)→(B,F,T), each sensor token has dim = timesteps
        timesteps = int(input_shape[1])
        ffn_hidden = max(timesteps * 2, self.ffn_dim)
        self.sensor_ffn = keras.Sequential(
            [
                layers.Dense(ffn_hidden, activation="gelu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(timesteps),  # output matches token dimension T
            ],
            name="sensor_ffn",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        # --- Stage 1: sensor-wise attention ---
        # Treat each feature/sensor as a token: transpose (B,T,F) → (B,F,T)
        x_sensor = tf.transpose(inputs, perm=[0, 2, 1])  # (B, F, T)
        attn1 = self.sensor_mha(x_sensor, x_sensor, training=training)
        x_sensor = self.sensor_norm1(x_sensor + attn1)
        ffn1 = self.sensor_ffn(x_sensor, training=training)
        x_sensor = self.sensor_norm2(x_sensor + ffn1)
        # Restore original layout
        x = tf.transpose(x_sensor, perm=[0, 2, 1])  # (B, T, F)

        # --- Stage 2: temporal attention ---
        attn2 = self.temporal_mha(x, x, training=training)
        x = self.temporal_norm1(x + attn2)
        ffn2 = self.temporal_ffn(x, training=training)
        x = self.temporal_norm2(x + ffn2)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


@ModelRegistry.register("star_transformer")
class STARTransformer(BaseModel):
    """STAR two-stage hierarchical transformer.

    Reference: Sensors 2024, 24(3), 824 — "A Two-Stage Attention-Based
    Hierarchical Transformer for Turbofan Engine Remaining Useful Life
    Prediction"
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_layers: int = 3,
        **kwargs,
    ) -> keras.Model:
        """Build and compile STAR Transformer.

        Args:
            input_shape: (timesteps, features)
            units: Model dimension d_model
            dense_units: Output dense layer size
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            num_heads: Number of attention heads (must divide units)
            num_layers: Number of stacked STARBlocks

        Returns:
            Compiled Keras model with output shape (batch, 1)
        """
        timesteps, features = input_shape

        inputs = layers.Input(shape=input_shape, name="input")

        # Project input features to model dimension
        x = layers.Dense(units, name="input_proj")(inputs)

        # Learnable positional embedding
        positions = tf.range(start=0, limit=timesteps, delta=1)
        pos_emb = layers.Embedding(
            input_dim=timesteps, output_dim=units, name="pos_embedding"
        )(positions)
        x = x + pos_emb  # broadcast over batch dimension

        x = layers.Dropout(dropout_rate, name="input_dropout")(x)

        # Stacked STAR blocks
        for i in range(num_layers):
            x = STARBlock(
                d_model=units,
                num_heads=num_heads,
                ffn_dim=units * 2,
                dropout=dropout_rate,
                name=f"star_block_{i}",
            )(x)

        # Aggregate temporal dimension → fixed-size representation
        x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
        x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
        x = layers.Dropout(dropout_rate, name="output_dropout")(x)
        outputs = layers.Dense(1, activation="linear", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="star_transformer")
        return compile_model_for_training(model, learning_rate=learning_rate)
