"""
Convolutional Neural Network architectures for RUL prediction.

Includes:
- TCN: Temporal Convolutional Network with dilated causal convolutions
- WaveNet: WaveNet-style gated activations with dilated convolutions
"""

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseModel
from .registry import ModelRegistry

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


class WaveNetBlock(layers.Layer):
    """WaveNet-style gated activation with dilated causal convolutions."""

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

        # Gated activation: conv_tanh * conv_sigmoid
        self.conv_tanh = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="tanh",
        )
        self.conv_sigmoid = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="sigmoid",
        )
        self.conv_out = layers.Conv1D(filters=filters, kernel_size=1)
        self.dropout = layers.Dropout(dropout_rate)
        self.residual_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv1D(filters=self.filters, kernel_size=1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Gated activation
        tanh_out = self.conv_tanh(inputs)
        sigmoid_out = self.conv_sigmoid(inputs)
        gated = layers.multiply([tanh_out, sigmoid_out])

        # Output projection
        x = self.conv_out(gated)
        x = self.dropout(x, training=training)

        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs

        return layers.add([x, residual])

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


@ModelRegistry.register("wavenet")
class WaveNetModel(BaseModel):
    """
    WaveNet-style model with gated dilated causal convolutions.
    Very effective for time series with long-range dependencies.
    Often outperforms RNNs on sequential data.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 2,
        num_layers: int = 8,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        x = inputs
        # Stack WaveNet blocks with exponentially increasing dilation
        for i in range(num_layers):
            dilation_rate = 2**i
            x = WaveNetBlock(
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



