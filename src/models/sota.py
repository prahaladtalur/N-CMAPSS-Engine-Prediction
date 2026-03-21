"""Wrappers for standalone SOTA architecture modules."""

from typing import Tuple

from tensorflow import keras

from src.models.atcn import build_atcn_model
from src.models.base import BaseModel
from src.models.cata_tcn import build_cata_tcn_model
from src.models.cnn_lstm_attention import build_cnn_lstm_attention_model
from src.models.mdfa import MDFAModule
from src.models.mstcn import build_mstcn_model
from src.models.registry import ModelRegistry
from src.models.sparse_transformer_bigrcu import build_sparse_transformer_bigrcu_model
from src.models.ttsnet import build_ttsnet_model
from tensorflow.keras import layers


@ModelRegistry.register("mdfa")
class MDFAModel(BaseModel):
    """Multi-Scale Dilated Fusion Attention model."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        dilation_rates: list = None,
    ) -> keras.Model:
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        inputs = layers.Input(shape=input_shape)
        x = MDFAModule(
            filters=units,
            dilation_rates=dilation_rates,
            kernel_size=3,
            dropout_rate=dropout_rate,
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mdfa")
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("cnn_lstm_attention")
class CNNLSTMAttentionModel(BaseModel):
    """CNN-LSTM-Attention wrapper."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 128,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        cnn_filters: list = None,
    ) -> keras.Model:
        if cnn_filters is None:
            cnn_filters = [64, 128, 256]
        return build_cnn_lstm_attention_model(
            input_shape=input_shape,
            cnn_filters=cnn_filters,
            lstm_units=units,
            attention_units=units // 2,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )


@ModelRegistry.register("cata_tcn")
class CATATCNModel(BaseModel):
    """Channel-and-temporal attention TCN wrapper."""

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
        return build_cata_tcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )


@ModelRegistry.register("ttsnet")
class TTSNetModel(BaseModel):
    """Transformer + TCN + self-attention fusion wrapper."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        kernel_size: int = 3,
    ) -> keras.Model:
        return build_ttsnet_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            kernel_size=kernel_size,
        )


@ModelRegistry.register("atcn")
class ATCNModel(BaseModel):
    """Attention-Based Temporal Convolutional Network wrapper."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        kernel_size: int = 3,
        num_tcn_layers: int = 4,
    ) -> keras.Model:
        return build_atcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_tcn_layers=num_tcn_layers,
        )


@ModelRegistry.register("sparse_transformer_bigrcu")
class SparseTransformerBiGRCUModel(BaseModel):
    """Sparse Transformer with Bi-GRCU ensemble wrapper."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        local_window: int = 32,
        num_global_tokens: int = 8,
    ) -> keras.Model:
        return build_sparse_transformer_bigrcu_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            local_window=local_window,
            num_global_tokens=num_global_tokens,
        )


@ModelRegistry.register("mstcn")
class MSTCNModel(BaseModel):
    """Multi-Scale Temporal Convolutional Network wrapper."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 3,
        dilation_rates: list = None,
    ) -> keras.Model:
        return build_mstcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            kernel_size=kernel_size,
            dilation_rates=dilation_rates,
        )
