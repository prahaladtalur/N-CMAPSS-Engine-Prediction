"""Attention-oriented architectures."""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel
from src.models.registry import ModelRegistry


class AttentionLayer(layers.Layer):
    """Simple additive attention over the time dimension."""

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
        scores = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        return super().get_config()


@ModelRegistry.register("attention_lstm")
class AttentionLSTMModel(BaseModel):
    """LSTM with additive attention."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(units, return_sequences=True)(inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(units // 2, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = AttentionLayer()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("transformer")
class TransformerModel(BaseModel):
    """Transformer encoder for sequence modeling."""

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
        x = layers.Dense(units)(inputs)

        for _ in range(num_layers):
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=units // num_heads
            )(x, x)
            attn_output = layers.Dropout(dropout_rate)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

            ffn_output = layers.Dense(units * 2, activation="relu")(x)
            ffn_output = layers.Dense(units)(ffn_output)
            ffn_output = layers.Dropout(dropout_rate)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("resnet_lstm")
class ResNetLSTMModel(BaseModel):
    """Stacked LSTMs with residual connections."""

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
        x = layers.Dense(units)(inputs)

        for i in range(num_layers):
            lstm_out = layers.LSTM(units, return_sequences=True)(x)
            lstm_out = layers.Dropout(dropout_rate)(lstm_out)
            x = layers.add([x, lstm_out]) if i > 0 else lstm_out
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        x = layers.LSTM(units // 2, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return BaseModel.compile_model(model, learning_rate)
