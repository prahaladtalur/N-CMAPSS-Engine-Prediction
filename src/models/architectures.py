"""
Model architectures for RUL prediction with easy switching via registry.

Includes SOTA RNN architectures:
- LSTM (baseline)
- Bidirectional LSTM
- GRU
- Bidirectional GRU
- Attention LSTM
- Temporal Convolutional Network (TCN)
"""

from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ModelRegistry:
    """Registry for easy model switching."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""

        def decorator(model_class):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def build(
        cls,
        name: str,
        input_shape: Tuple[int, int],
        **kwargs,
    ) -> keras.Model:
        """Build a model by name with given configuration."""
        model_class = cls.get(name)
        return model_class.build(input_shape, **kwargs)


def asymmetric_mse(alpha: float = 2.0):
    """
    Asymmetric MSE loss that penalizes late RUL predictions more heavily.

    In RUL prediction, over-predicting remaining life (y_pred > y_true) is
    dangerous — it risks operating past failure.  This loss applies a penalty
    multiplier of ``alpha`` to squared errors when the prediction exceeds the
    true value, while standard squared error is used for early predictions.

    Args:
        alpha: Penalty multiplier for late predictions (default 2.0).
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

    return loss


class BaseModel(ABC):
    """Base class for all models."""

    @staticmethod
    @abstractmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        """Build and compile the model."""
        pass

    @staticmethod
    def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
        """Compile model with standard settings."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=asymmetric_mse(),
            metrics=["mae", "mape"],
        )
        return model


@ModelRegistry.register("lstm")
class LSTMModel(BaseModel):
    """Standard LSTM model for RUL prediction."""

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
                layers.LSTM(units, return_sequences=True),
                layers.Dropout(dropout_rate),
                layers.LSTM(units // 2, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("bilstm")
class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model - captures both past and future context."""

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
                layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
                layers.Dropout(dropout_rate),
                layers.Bidirectional(layers.LSTM(units // 2, return_sequences=False)),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("gru")
class GRUModel(BaseModel):
    """GRU model - simpler than LSTM, often faster with similar performance."""

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
                layers.GRU(units, return_sequences=True),
                layers.Dropout(dropout_rate),
                layers.GRU(units // 2, return_sequences=False),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("bigru")
class BiGRUModel(BaseModel):
    """Bidirectional GRU model."""

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
                layers.Bidirectional(layers.GRU(units, return_sequences=True)),
                layers.Dropout(dropout_rate),
                layers.Bidirectional(layers.GRU(units // 2, return_sequences=False)),
                layers.Dropout(dropout_rate),
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


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


@ModelRegistry.register("mlp")
class MLPModel(BaseModel):
    """
    Simple Multi-Layer Perceptron baseline.
    Flattens time series and uses only dense layers.
    Useful as a baseline to compare against temporal models.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_hidden_layers: int = 3,
    ) -> keras.Model:
        model = keras.Sequential([layers.Input(shape=input_shape)])

        # Flatten the time series
        model.add(layers.Flatten())

        # Hidden layers
        for i in range(num_hidden_layers):
            layer_units = units // (2**i) if i > 0 else units
            model.add(layers.Dense(layer_units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))

        # Final dense layer
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation="linear"))

        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("cnn_gru")
class CNNGRUModel(BaseModel):
    """
    CNN-GRU hybrid - CNN for feature extraction, GRU for temporal modeling.
    Similar to CNN-LSTM but often faster with comparable performance.
    """

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
                # GRU for temporal modeling
                layers.GRU(units, return_sequences=False),
                layers.Dropout(dropout_rate),
                # Dense layers
                layers.Dense(dense_units, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation="linear"),
            ]
        )
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("inception_lstm")
class InceptionLSTMModel(BaseModel):
    """
    Inception-style multi-scale feature extraction followed by LSTM.
    Captures patterns at different time scales simultaneously.
    Good for complex time series with multi-scale patterns.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)

        # Inception module - parallel convolutions with different kernel sizes
        conv1 = layers.Conv1D(filters=16, kernel_size=1, activation="relu", padding="same")(inputs)

        conv3 = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)

        conv5 = layers.Conv1D(filters=16, kernel_size=5, activation="relu", padding="same")(inputs)

        pool = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(inputs)
        pool = layers.Conv1D(filters=16, kernel_size=1, activation="relu", padding="same")(pool)

        # Concatenate all parallel paths
        x = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
        x = layers.Dropout(dropout_rate)(x)

        # LSTM layers
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units // 2, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Dense layers
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


def get_model(
    model_name: str,
    input_shape: Tuple[int, int],
    **kwargs,
) -> keras.Model:
    """
    Get a model by name with given configuration.

    Args:
        model_name: Name of the model (lstm, bilstm, gru, bigru, attention_lstm, tcn, cnn_lstm, transformer)
        input_shape: Shape of input data (timesteps, features)
        **kwargs: Model-specific arguments (units, dense_units, dropout_rate, learning_rate)

    Returns:
        Compiled Keras model
    """
    return ModelRegistry.build(model_name, input_shape, **kwargs)


def list_available_models() -> list:
    """List all available model architectures."""
    return ModelRegistry.list_models()


def get_model_info() -> Dict[str, str]:
    """Get descriptions of all available models."""
    return {
        # RNN-based models
        "lstm": "Standard LSTM - baseline RNN for sequence modeling",
        "bilstm": "Bidirectional LSTM - captures both past and future context",
        "gru": "GRU - simpler than LSTM, often faster with similar performance",
        "bigru": "Bidirectional GRU - bidirectional version of GRU",
        "attention_lstm": "LSTM with Attention - focuses on important timesteps",
        "resnet_lstm": "ResNet-LSTM - residual connections for better gradient flow",
        # Convolutional models
        "tcn": "Temporal Convolutional Network - SOTA, parallelizable, large receptive field",
        "wavenet": "WaveNet - gated dilated convolutions, excellent for long sequences",
        # Hybrid models
        "cnn_lstm": "CNN-LSTM hybrid - CNN extracts features, LSTM models time",
        "cnn_gru": "CNN-GRU hybrid - CNN + GRU, faster than CNN-LSTM",
        "inception_lstm": "Inception-LSTM - multi-scale feature extraction",
        # Attention-based models
        "transformer": "Transformer encoder - self-attention based, very SOTA",
        # Baseline
        "mlp": "Simple MLP - baseline for comparison (no temporal modeling)",
    }


def get_model_recommendations() -> Dict[str, list]:
    """Get model recommendations for different use cases."""
    return {
        "quick_baseline": ["mlp", "gru"],
        "best_accuracy": ["transformer", "attention_lstm", "wavenet", "resnet_lstm"],
        "fastest_training": ["gru", "cnn_gru", "tcn"],
        "most_interpretable": ["lstm", "attention_lstm"],
        "long_sequences": ["tcn", "wavenet", "transformer"],
        "limited_data": ["gru", "lstm"],
        "complex_patterns": ["transformer", "inception_lstm", "wavenet"],
    }
