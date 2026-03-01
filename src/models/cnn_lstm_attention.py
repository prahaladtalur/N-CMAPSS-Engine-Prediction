"""
CNN-LSTM-Attention architecture for RUL prediction.

Based on the paper:
"Prediction of Remaining Useful Life of Aero-engines Based on CNN-LSTM-Attention" (2024)
https://link.springer.com/article/10.1007/s44196-024-00639-w

Architecture:
1. CNN layers for local feature extraction
2. Stacked LSTM for temporal modeling
3. Self-attention mechanism to focus on important timesteps
4. Dense layers for RUL prediction

Achieves RMSE 13.907-16.637 on CMAPSS datasets.
"""

import tensorflow as tf
from tensorflow.keras import layers


class SelfAttentionLayer(layers.Layer):
    """
    Self-attention mechanism for sequence data.

    Computes attention scores over the sequence to identify which timesteps
    are most relevant for the prediction task. Different from the attention
    in MDFA, this operates on the LSTM output sequence.

    Architecture:
        LSTM outputs (B, T, D) → Query/Key/Value projections
        → Scaled dot-product attention → Context vector (B, D)
    """

    def __init__(self, units: int, **kwargs):
        """
        Args:
            units: Dimension for Query, Key, Value projections
        """
        super().__init__(**kwargs)
        self.units = units

        # Query, Key, Value projection layers
        self.W_q = layers.Dense(units, use_bias=False)
        self.W_k = layers.Dense(units, use_bias=False)
        self.W_v = layers.Dense(units, use_bias=False)

    def call(self, inputs):
        """
        Args:
            inputs: LSTM outputs (batch_size, timesteps, features)

        Returns:
            Attention-weighted context vector (batch_size, features)
        """
        # Project to Query, Key, Value
        Q = self.W_q(inputs)  # (B, T, units)
        K = self.W_k(inputs)  # (B, T, units)
        V = self.W_v(inputs)  # (B, T, units)

        # Scaled dot-product attention
        # Attention scores: Q * K^T / sqrt(d_k)
        scores = tf.matmul(Q, K, transpose_b=True)  # (B, T, T)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, T, T)

        # Apply attention to values
        context = tf.matmul(attention_weights, V)  # (B, T, units)

        # Take the last timestep's context (or could average/max pool)
        # Using last timestep aligns with the LSTM's final state
        context = context[:, -1, :]  # (B, units)

        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class CNNFeatureExtractor(layers.Layer):
    """
    CNN-based feature extractor for time series data.

    Uses multiple Conv1D layers with increasing dilation and pooling
    to extract hierarchical local features from sensor readings.

    Architecture:
        Conv1D (64 filters) → BatchNorm → ReLU → MaxPool
        → Conv1D (128 filters) → BatchNorm → ReLU → MaxPool
        → Conv1D (256 filters) → BatchNorm → ReLU
    """

    def __init__(
        self, filters_list: list = None, kernel_size: int = 3, dropout_rate: float = 0.2, **kwargs
    ):
        """
        Args:
            filters_list: List of filter counts for each conv layer (default: [64, 128, 256])
            kernel_size: Kernel size for convolutions (default: 3)
            dropout_rate: Dropout rate after each conv block (default: 0.2)
        """
        super().__init__(**kwargs)

        if filters_list is None:
            filters_list = [64, 128, 256]

        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Build conv blocks
        self.conv_blocks = []
        for i, filters in enumerate(filters_list):
            block = {
                "conv": layers.Conv1D(
                    filters=filters, kernel_size=kernel_size, padding="same", activation=None
                ),
                "batch_norm": layers.BatchNormalization(),
                "activation": layers.Activation("relu"),
                "dropout": layers.Dropout(dropout_rate),
            }

            # Add pooling for first two blocks to downsample
            if i < 2:
                block["pool"] = layers.MaxPooling1D(pool_size=2, padding="same")

            self.conv_blocks.append(block)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Input tensor (batch_size, timesteps, features)

        Returns:
            Extracted features (batch_size, downsampled_timesteps, filters)
        """
        x = inputs

        for block in self.conv_blocks:
            x = block["conv"](x)
            x = block["batch_norm"](x, training=training)
            x = block["activation"](x)
            x = block["dropout"](x, training=training)

            if "pool" in block:
                x = block["pool"](x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters_list": self.filters_list,
                "kernel_size": self.kernel_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def build_cnn_lstm_attention_model(
    input_shape: tuple,
    cnn_filters: list = None,
    lstm_units: int = 128,
    attention_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build CNN-LSTM-Attention model for RUL prediction.

    Architecture from the 2024 paper:
    "Prediction of Remaining Useful Life of Aero-engines Based on CNN-LSTM-Attention"

    Args:
        input_shape: (timesteps, features) - e.g., (1000, 32) for N-CMAPSS
        cnn_filters: List of filter counts for CNN layers (default: [64, 128, 256])
        lstm_units: Number of LSTM units (default: 128)
        attention_units: Dimension for attention mechanism (default: 64)
        dense_units: Units in final dense layer (default: 32)
        dropout_rate: Dropout rate (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_cnn_lstm_attention_model(input_shape=(1000, 32))
        >>> model.summary()
    """
    if cnn_filters is None:
        cnn_filters = [64, 128, 256]

    inputs = layers.Input(shape=input_shape)

    # 1. CNN Feature Extraction
    cnn_features = CNNFeatureExtractor(
        filters_list=cnn_filters, kernel_size=3, dropout_rate=dropout_rate
    )(inputs)

    # 2. Stacked LSTM for temporal modeling
    # First LSTM layer (returns sequences for attention)
    x = layers.LSTM(lstm_units, return_sequences=True)(cnn_features)
    x = layers.Dropout(dropout_rate)(x)

    # Second LSTM layer (returns sequences for attention)
    x = layers.LSTM(lstm_units // 2, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)

    # 3. Self-Attention Mechanism
    # Attention focuses on important timesteps
    attention_output = SelfAttentionLayer(units=attention_units)(x)

    # 4. Dense layers for final prediction
    x = layers.Dense(dense_units, activation="relu")(attention_output)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_attention")

    # Compile with asymmetric MSE loss (penalizes late predictions more)
    def asymmetric_mse(alpha: float = 2.0):
        def loss(y_true, y_pred):
            error = y_pred - y_true
            return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

        return loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=asymmetric_mse(), metrics=["mae", "mape"])

    return model
