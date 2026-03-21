"""Convolutional and temporal convolutional architectures."""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel
from src.models.registry import ModelRegistry


class TCNBlock(layers.Layer):
    """Residual dilated causal convolution block."""

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
            self.downsample = layers.Conv1D(filters=self.filters, kernel_size=1, padding="same")
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        residual = self.downsample(inputs) if self.downsample is not None else inputs
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
    """Temporal Convolutional Network."""

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
        for i in range(num_layers):
            x = TCNBlock(
                filters=units,
                kernel_size=kernel_size,
                dilation_rate=2**i,
                dropout_rate=dropout_rate,
            )(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


class WaveNetBlock(layers.Layer):
    """WaveNet-style gated residual block."""

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
        gated = layers.multiply([self.conv_tanh(inputs), self.conv_sigmoid(inputs)])
        x = self.conv_out(gated)
        x = self.dropout(x, training=training)
        residual = self.residual_conv(inputs) if self.residual_conv is not None else inputs
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
    """WaveNet-style model with gated dilated convolutions."""

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
        for i in range(num_layers):
            x = WaveNetBlock(
                filters=units,
                kernel_size=kernel_size,
                dilation_rate=2**i,
                dropout_rate=dropout_rate,
            )(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)
