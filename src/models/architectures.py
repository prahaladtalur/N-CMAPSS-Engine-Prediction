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

from typing import Tuple, Dict, Any, Callable, Optional
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
            loss="mse",
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
    """Custom attention layer for sequence models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        # Weighted sum
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
    """Temporal Convolutional Network block with dilated causal convolutions."""

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

        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
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
        self.downsample = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                filters=self.filters, kernel_size=1, padding="same"
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(inputs)
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
        "lstm": "Standard LSTM - baseline RNN for sequence modeling",
        "bilstm": "Bidirectional LSTM - captures both past and future context",
        "gru": "GRU - simpler than LSTM, often faster with similar performance",
        "bigru": "Bidirectional GRU - bidirectional version of GRU",
        "attention_lstm": "LSTM with Attention - focuses on important timesteps",
        "tcn": "Temporal Convolutional Network - SOTA, parallelizable, large receptive field",
        "cnn_lstm": "CNN-LSTM hybrid - CNN extracts features, LSTM models time",
        "transformer": "Transformer encoder - self-attention based, very SOTA",
    }
