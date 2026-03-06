"""
Attention-based architectures for RUL prediction.

Includes:
- AttentionLSTM: LSTM with attention mechanism
- Transformer: Multi-head self-attention encoder
- ResNetLSTM: LSTM with residual connections
"""

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseModel
from .registry import ModelRegistry

class AttentionLayer(layers.Layer):
    """
    Custom attention layer for sequence models.

    Implements a simple additive attention mechanism that learns to focus on
    important timesteps in the input sequence. The attention weights are computed
    using a learned linear transformation followed by tanh activation and softmax.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Learnable attention weights: (feature_dim, 1) for scoring each feature
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Learnable bias: (sequence_length, 1) for position-dependent scoring
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores: e = tanh(W^T * x + b)
        # This gives a score for each timestep indicating its importance
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Normalize scores to get attention weights (probabilities over timesteps)
        a = tf.nn.softmax(e, axis=1)
        # Compute weighted sum: output = sum(attention_weights * inputs)
        # This aggregates the sequence into a single vector, weighted by importance
        output = tf.reduce_sum(inputs * a, axis=1)
        return output

    def get_config(self):
        return super().get_config()


@ModelRegistry.register("attention_lstm")
class AttentionLSTMModel(BaseModel):
    """LSTM with attention mechanism - SOTA for many sequence tasks."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        # LSTM layers
        x = layers.LSTM(units, return_sequences=True)(inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units // 2, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Attention layer
        x = AttentionLayer()(x)

        # Dense layers
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Network block with dilated causal convolutions.

    TCN blocks use dilated convolutions to capture long-range dependencies without
    the sequential processing overhead of RNNs. Each block consists of two dilated
    causal convolutions with residual connections for better gradient flow.

    Key features:
    - Causal padding: ensures no future information leaks into past predictions
    - Dilated convolutions: exponentially increase receptive field (2^layer_depth)
    - Residual connections: help with training deep networks
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # Two dilated causal convolutions in sequence
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",  # No future information leakage
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.downsample = None  # Will be created if input dimension doesn't match

    def build(self, input_shape):
        # If input feature dimension doesn't match output, add 1x1 conv for residual
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(filters=self.filters, kernel_size=1, padding="same")
        super().build(input_shape)

    def call(self, inputs, training=None):
        # First dilated convolution
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        # Second dilated convolution
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        # Residual connection: helps with gradient flow and training stability
        if self.downsample is not None:
            residual = self.downsample(inputs)  # Project input to match dimensions
        else:
            residual = inputs  # Direct connection if dimensions match

        return layers.add([x, residual])  # Element-wise addition

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@ModelRegistry.register("tcn")
class TCNModel(BaseModel):
    """
    Temporal Convolutional Network - SOTA for many sequence modeling tasks.
    Uses dilated causal convolutions with residual connections.
    Often outperforms RNNs while being more parallelizable.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 3,
        num_layers: int = 4,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        x = inputs
        # Stack TCN blocks with exponentially increasing dilation
        for i in range(num_layers):
            dilation_rate = 2**i
            x = TCNBlock(
                filters=units,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
            )(x)

        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("cnn_lstm")
class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid - CNN extracts features, LSTM models temporal dependencies."""

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
                # CNN feature extraction
                layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(dropout_rate),
                # LSTM for temporal modeling
                layers.LSTM(units, return_sequences=False),
                layers.Dropout(dropout_rate),
                # Dense layers
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("transformer")
class TransformerModel(BaseModel):
    """
    Transformer encoder for sequence modeling.
    Uses self-attention mechanism - very SOTA for sequence tasks.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        # Positional encoding (simple learnable)
        x = layers.Dense(units)(inputs)

        # Transformer encoder blocks
        for _ in range(num_layers):
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=units // num_heads
            )(x, x)
            attn_output = layers.Dropout(dropout_rate)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed-forward network
            ffn_output = layers.Dense(units * 2, activation="relu")(x)
            ffn_output = layers.Dense(units)(ffn_output)
            ffn_output = layers.Dropout(dropout_rate)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Global average pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("resnet_lstm")
class ResNetLSTMModel(BaseModel):
    """
    ResNet-style LSTM with residual connections for better gradient flow.
    Helps prevent vanishing gradients in deep networks.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_layers: int = 3,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        # Initial projection to match LSTM output dimension
        x = layers.Dense(units)(inputs)

        # Stacked LSTM layers with residual connections
        for i in range(num_layers):
            lstm_out = layers.LSTM(units, return_sequences=True)(x)
            lstm_out = layers.Dropout(dropout_rate)(lstm_out)

            # Residual connection
            if i > 0:
                x = layers.add([x, lstm_out])
            else:
                x = lstm_out

            # Layer normalization for stability
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Final LSTM without residual
        x = layers.LSTM(units // 2, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Dense layers
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


