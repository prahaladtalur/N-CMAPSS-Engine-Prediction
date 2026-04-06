"""Sensor-wise attention model (iTransformer-style) + MSTCN backbone."""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel, compile_model_for_training
from src.models.cata_tcn import ChannelAttention1D, ResidualTCNBlock, TemporalAttention1D
from src.models.mstcn import GlobalFusionAttention  # reuse existing GFA
from src.models.registry import ModelRegistry


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class SensorWiseAttention(layers.Layer):
    """Apply multi-head self-attention across the sensor/feature dimension (iTransformer style).

    Transposes the input from (batch, timesteps, features) to (batch, features, timesteps),
    applies multi-head self-attention treating each sensor as a token attending to all other
    sensors, then transposes back to (batch, timesteps, features).

    The residual connection is applied in the transposed space before restoring the original
    layout.
    """

    def __init__(self, num_heads: int = 4, key_dim: int = 16, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # MHA is built lazily in build() so output_shape can match the feature dim
        self.mha = None
        self.dropout_layer = layers.Dropout(dropout)

    def build(self, input_shape: Tuple):
        # input_shape: (batch, timesteps, features)
        # After transpose, tokens are features and each token has dimension timesteps
        # We need output_shape=timesteps so the residual add works directly
        timesteps = int(input_shape[1])
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            output_shape=timesteps,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        # inputs: (B, T, F) — transpose so features become tokens
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (B, F, T)
        x = self.norm(x)
        attn_out = self.mha(x, x, training=training)  # (B, F, T) — self-attention across sensors
        attn_out = self.dropout_layer(attn_out, training=training)
        x = x + attn_out  # residual in (B, F, T) space
        return tf.transpose(x, perm=[0, 2, 1])  # back to (B, T, F)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


@ModelRegistry.register("sensor_attn_mstcn")
class SensorAttentionMSTCN(BaseModel):
    """MSTCN with iTransformer-style sensor-wise attention pre-processing.

    Architecture:
      1. SensorWiseAttention: multi-head attention across sensor/feature dimension
      2. Conv1D projection to model dimension
      3. Multi-scale TCN branches (4 dilation rates, 2 blocks each)
      4. GlobalFusionAttention to fuse scales
      5. GlobalAveragePooling -> Dense -> RUL output
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        dilation_rates: List[int] = None,
        **kwargs,
    ) -> keras.Model:
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        num_features = input_shape[-1]
        key_dim = max(num_features // 4, 4)

        inputs = layers.Input(shape=input_shape, name="input")

        # Stage 1: Sensor-wise attention (iTransformer style)
        x = SensorWiseAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name="sensor_wise_attention",
        )(inputs)

        # Stage 2: Project to model dimension
        x = layers.Conv1D(units, kernel_size=1, padding="same", activation="relu", name="proj")(x)

        # Stage 3: Multi-scale TCN branches
        scale_outputs = []
        for i, dilation in enumerate(dilation_rates):
            branch = ResidualTCNBlock(
                filters=units,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=dropout_rate,
                name=f"tcn_scale{i}_block1",
            )(x)
            branch = ResidualTCNBlock(
                filters=units,
                kernel_size=3,
                dilation_rate=dilation,
                dropout_rate=dropout_rate,
                name=f"tcn_scale{i}_block2",
            )(branch)
            scale_outputs.append(branch)

        # Stage 4: Global Fusion Attention
        fused = GlobalFusionAttention(
            num_scales=len(dilation_rates),
            reduction_ratio=8,
            name="global_fusion",
        )(scale_outputs)

        # Output head
        out = layers.GlobalAveragePooling1D(name="global_pooling")(fused)
        out = layers.Dense(dense_units, activation="relu", name="dense_1")(out)
        out = layers.Dropout(dropout_rate, name="dropout")(out)
        outputs = layers.Dense(1, activation="linear", name="output")(out)

        model = keras.Model(inputs=inputs, outputs=outputs, name="sensor_attn_mstcn")
        return compile_model_for_training(model, learning_rate=learning_rate)
