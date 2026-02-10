"""
Multi-Scale Dilated Fusion Attention (MDFA) components.

Implements the MDFA architecture from:
"Remaining Useful Life Prediction for Aero-Engines Based on Multi-Scale Dilated Fusion Attention Model"
https://www.mdpi.com/2076-3417/15/17/9813

Key components:
- Channel Attention (SENet-style): learns which sensor features are most important
- Spatial Attention: focuses on critical time windows in the degradation process
- Multi-scale Dilated Convolutions: captures patterns at different temporal scales
- Fusion mechanism: combines multi-scale features with attention weighting
"""

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """
    Channel Attention mechanism (SENet-style).

    Learns to emphasize important sensor features while suppressing noise.
    Uses global pooling + FC layers to compute channel-wise attention weights.

    Architecture:
        Input (B, T, C) → GlobalAvgPool → Dense(C/r) → ReLU → Dense(C) → Sigmoid → (B, C)
        Weights are broadcast and multiplied with input: (B, T, C) * (B, 1, C)
    """

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        """
        Args:
            reduction_ratio: Compression ratio for bottleneck layer (default: 8)
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        # Bottleneck layer for dimensionality reduction
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation="relu")
        # Expand back to original channel count
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        # Global average pooling: (B, T, C) → (B, C)
        pooled = tf.reduce_mean(inputs, axis=1)
        # Squeeze and excitation: (B, C) → (B, C/r) → (B, C)
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        # Broadcast and scale: (B, T, C) * (B, 1, C)
        return inputs * tf.expand_dims(weights, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention mechanism.

    Learns to focus on critical time windows in the sequence (e.g., when degradation accelerates).
    Uses channel-wise pooling + convolution to compute spatial attention weights.

    Architecture:
        Input (B, T, C) → [AvgPool, MaxPool] along C → Concat → Conv1D → Sigmoid → (B, T, 1)
        Weights are multiplied with input: (B, T, C) * (B, T, 1)
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        """
        Args:
            kernel_size: Convolution kernel size for spatial attention (default: 7)
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(
            filters=1, kernel_size=kernel_size, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        # Channel-wise pooling: (B, T, C) → (B, T, 1)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        # Concatenate pooled features: (B, T, 2)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        # Convolution to learn spatial weights: (B, T, 2) → (B, T, 1)
        weights = self.conv(concat)
        # Scale input by spatial weights: (B, T, C) * (B, T, 1)
        return inputs * weights

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class MDFAModule(layers.Layer):
    """
    Multi-Scale Dilated Fusion Attention (MDFA) module.

    Core component of the MDFA architecture that captures multi-scale temporal patterns
    through parallel dilated convolutions, then applies channel and spatial attention
    to emphasize informative features.

    Architecture:
        1. Parallel dilated convolutions with rates [1, 2, 4, 8] for multi-scale features
        2. Global pooling branch to capture sequence-level degradation trends
        3. Concatenate all branches
        4. Apply channel attention (which sensors matter)
        5. Apply spatial attention (when degradation accelerates)
        6. 1x1 convolution to fuse features

    Receptive fields with kernel_size=3:
        - Dilation 1: 3 timesteps (local patterns)
        - Dilation 2: 5 timesteps
        - Dilation 4: 9 timesteps
        - Dilation 8: 17 timesteps (long-term degradation)
    """

    def __init__(
        self,
        filters: int = 64,
        dilation_rates: list = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            filters: Number of filters for each dilated convolution branch
            dilation_rates: List of dilation rates for multi-scale branches (default: [1, 2, 4, 8])
            kernel_size: Kernel size for dilated convolutions (default: 3)
            dropout_rate: Dropout rate after convolutions (default: 0.2)
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates or [1, 2, 4, 8]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Parallel dilated convolution branches
        self.dilated_convs = [
            layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=rate,
                padding="same",
                activation="relu",
            )
            for rate in self.dilation_rates
        ]

        # Batch normalization for each branch
        self.batch_norms = [layers.BatchNormalization() for _ in self.dilation_rates]

        # Global pooling branch (captures sequence-level trends)
        self.global_pool = layers.GlobalAveragePooling1D()

        # Fusion convolution (1x1 conv to reduce concatenated features)
        # Output channels = filters after concatenating all branches
        num_branches = len(self.dilation_rates) + 1  # +1 for global pooling
        self.fusion_conv = layers.Conv1D(filters=filters, kernel_size=1, activation="relu")

        # Attention mechanisms
        self.channel_attention = ChannelAttention(reduction_ratio=8)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Dropout for regularization
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Multi-scale dilated convolutions
        conv_outputs = []
        for conv, bn in zip(self.dilated_convs, self.batch_norms):
            x = conv(inputs)
            x = bn(x, training=training)
            conv_outputs.append(x)

        # Global pooling branch: (B, T, C) → (B, C) → (B, 1, C) → (B, T, C)
        global_features = self.global_pool(inputs)
        # Broadcast to match sequence length
        global_features = tf.expand_dims(global_features, axis=1)
        global_features = tf.tile(global_features, [1, tf.shape(inputs)[1], 1])
        conv_outputs.append(global_features)

        # Concatenate all branches: (B, T, filters * num_branches)
        concatenated = tf.concat(conv_outputs, axis=-1)

        # Fusion via 1x1 convolution: (B, T, filters * num_branches) → (B, T, filters)
        fused = self.fusion_conv(concatenated)

        # Apply channel attention
        fused = self.channel_attention(fused)

        # Apply spatial attention
        fused = self.spatial_attention(fused)

        # Dropout
        fused = self.dropout(fused, training=training)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "dilation_rates": self.dilation_rates,
                "kernel_size": self.kernel_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
