"""
Multi-Scale Temporal Convolutional Network (MSTCN) variants.

This module keeps the original repo MSTCN implementation for backwards
compatibility and adds a paper-shaped SA-MSTCN-GFA variant that more closely
matches the 2024 architecture:

    Input → Self-Attention → 3 parallel TCN branches (different kernel scales)
          → Global Fusion Attention → Dense layers → RUL
"""

from __future__ import annotations

from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.cata_tcn import ResidualTCNBlock, ChannelAttention1D, TemporalAttention1D


def asymmetric_mse(alpha: float = 2.0):
    """Penalize late predictions more than early ones."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

    return loss


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class GlobalFusionAttention(layers.Layer):
    """
    Global Fusion Attention mechanism for multi-scale feature integration.

    Combines three types of attention to intelligently fuse multi-scale features:
    1. Channel attention: identifies which sensors/features are most important
    2. Temporal attention: highlights critical time windows for degradation
    3. Cross-scale attention: learns how to weight different dilation scales

    Unlike simple concatenation (MDFA), GFA uses adaptive gating to suppress
    redundant information across scales.
    """

    def __init__(self, num_scales: int = 4, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.reduction_ratio = reduction_ratio

        # Channel attention for each scale
        self.channel_attentions = [
            ChannelAttention1D(reduction_ratio=reduction_ratio, name=f"channel_attn_{i}")
            for i in range(num_scales)
        ]

        # Temporal attention for each scale
        self.temporal_attentions = [
            TemporalAttention1D(kernel_size=7, name=f"temporal_attn_{i}")
            for i in range(num_scales)
        ]

        # Cross-scale fusion weights (learnable)
        self.scale_weights = None
        self.fusion_gate = None

    def build(self, input_shapes: List[Tuple[int, ...]]):
        # input_shapes is a list of shapes, one per scale
        # Each shape: (batch_size, timesteps, features)

        # Learnable scale importance weights
        self.scale_weights = self.add_weight(
            name="scale_weights",
            shape=(self.num_scales,),
            initializer="ones",
            trainable=True,
        )

        # Adaptive gating network for redundancy suppression
        # Takes concatenated features and outputs gate values
        if len(input_shapes) > 0:
            total_channels = sum(shape[-1] for shape in input_shapes)
            self.fusion_gate = keras.Sequential(
                [
                    layers.Dense(total_channels // self.reduction_ratio, activation="relu"),
                    layers.Dense(total_channels, activation="sigmoid"),
                ],
                name="fusion_gate",
            )

        super().build(input_shapes)

    def call(self, multi_scale_features: List[tf.Tensor]) -> tf.Tensor:
        """
        Fuse multi-scale features with global attention.

        Args:
            multi_scale_features: List of tensors from different scales
                                 Each tensor: (batch_size, timesteps, features)

        Returns:
            Fused features: (batch_size, timesteps, total_features)
        """
        # Apply channel and temporal attention to each scale
        attended_features = []
        for i, features in enumerate(multi_scale_features):
            # Channel attention: reweight feature channels
            x = self.channel_attentions[i](features)

            # Temporal attention: highlight important timesteps
            x = self.temporal_attentions[i](x)

            # Apply learned scale weight
            scale_weight = self.scale_weights[i]
            x = x * scale_weight

            attended_features.append(x)

        # Concatenate all scales
        concatenated = tf.concat(attended_features, axis=-1)

        # Apply adaptive gating to suppress redundancy
        # Global pooling for gate computation
        pooled = tf.reduce_mean(concatenated, axis=1)  # (B, total_features)
        gate = self.fusion_gate(pooled)  # (B, total_features)

        # Broadcast gate to timesteps and apply
        gate = tf.expand_dims(gate, axis=1)  # (B, 1, total_features)
        fused = concatenated * gate

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({"num_scales": self.num_scales, "reduction_ratio": self.reduction_ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class SequenceSelfAttention(layers.Layer):
    """Sequence-preserving self-attention block for SA-MSTCN-GFA."""

    def __init__(self, num_heads: int = 4, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
        self.position_embedding = None
        self.mha = None
        self.add = layers.Add()

    def build(self, input_shape: Tuple[int, ...]):
        seq_length = input_shape[1]
        feature_dim = input_shape[2]
        key_dim = max(feature_dim // self.num_heads, 1)
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(seq_length, feature_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            output_shape=feature_dim,
            name="multi_head_attention",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs + self.position_embedding
        x = self.layer_norm(x)
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        return self.add([inputs, attn_output])

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "dropout_rate": self.dropout_rate})
        return config


def build_mstcn_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    kernel_size: int = 3,
    dilation_rates: List[int] = None,
) -> keras.Model:
    """
    Build MSTCN model for RUL prediction.

    Args:
        input_shape: (sequence_length, num_features)
        units: Number of filters in TCN blocks
        dense_units: Number of units in final dense layer
        dropout_rate: Dropout probability
        learning_rate: Optimizer learning rate
        kernel_size: Kernel size for TCN convolutions
        dilation_rates: List of dilation rates for multi-scale branches
                       (default: [1, 2, 4, 8])

    Returns:
        Compiled Keras model
    """
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    num_scales = len(dilation_rates)

    inputs = layers.Input(shape=input_shape, name="input")

    # Self-Attention at head to emphasize critical timesteps
    # Note: SelfAttentionLayer from CNN-LSTM-Attention outputs (B, units)
    # We need to keep sequence dimension, so we'll use it differently or skip it
    # For MSTCN, let's apply attention after TCN instead

    # Multi-Scale TCN: parallel branches with different dilation rates
    tcn_outputs = []
    for i, dilation_rate in enumerate(dilation_rates):
        # Each scale has 2 stacked TCN blocks with the same dilation
        branch = inputs

        # First TCN block
        branch = ResidualTCNBlock(
            filters=units,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            name=f"tcn_scale{i}_block1",
        )(branch)

        # Second TCN block (same dilation for deeper receptive field)
        branch = ResidualTCNBlock(
            filters=units,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            name=f"tcn_scale{i}_block2",
        )(branch)

        tcn_outputs.append(branch)

    # Global Fusion Attention: intelligently combine multi-scale features
    fused = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8, name="global_fusion")(
        tcn_outputs
    )

    # Global pooling to get fixed-size representation
    x = layers.GlobalAveragePooling1D(name="global_pooling")(fused)

    # Dense output layers
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mstcn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"],
    )
    return model


def build_sa_mstcn_gfa_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    branch_kernel_sizes: List[int] = None,
    branch_dilations: List[int] = None,
    num_heads: int = 4,
) -> keras.Model:
    """Build the paper-shaped SA-MSTCN-GFA model.

    The key differences vs the legacy repo MSTCN are:
    - a real self-attention block at the head
    - three parallel branches with different kernel sizes for short/medium/long scales
    - a shared short dilation stack inside each branch
    """
    if branch_kernel_sizes is None:
        branch_kernel_sizes = [3, 5, 7]
    if branch_dilations is None:
        branch_dilations = [1, 2]

    inputs = layers.Input(shape=input_shape, name="input")
    x = SequenceSelfAttention(
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        name="head_self_attention",
    )(inputs)

    tcn_outputs = []
    for branch_idx, branch_kernel_size in enumerate(branch_kernel_sizes):
        branch = x
        for block_idx, dilation_rate in enumerate(branch_dilations, start=1):
            branch = ResidualTCNBlock(
                filters=units,
                kernel_size=branch_kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
                name=f"mstcn_branch{branch_idx}_block{block_idx}",
            )(branch)
        tcn_outputs.append(branch)

    fused = GlobalFusionAttention(
        num_scales=len(branch_kernel_sizes),
        reduction_ratio=8,
        name="global_fusion",
    )(tcn_outputs)
    fused = layers.Conv1D(
        filters=units,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="fusion_projection",
    )(fused)

    x = layers.GlobalAveragePooling1D(name="global_pooling")(fused)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sa_mstcn_gfa")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_mse(),
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"],
    )
    return model
