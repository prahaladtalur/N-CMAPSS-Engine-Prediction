"""Mamba-inspired Selective State Space Model for RUL prediction.

Implements a simplified but faithful Mamba SSM architecture using Keras
primitives. The key ideas from Mamba (Gu & Dao, 2023):
  1. Input-dependent (selective) state transitions via learned Δ, B, C
  2. Gated architecture with SiLU activation
  3. Linear O(N) complexity through sequential state updates

Architecture:
  Input → Conv1D projection → N × MambaBlocks → GlobalAvgPool → Dense → RUL

Since TF/Keras lacks a hardware-optimized parallel scan, the SSM recurrence
is implemented as a custom RNN cell (correct; not maximally fast but portable).

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           Gu & Dao, 2023 (https://arxiv.org/abs/2312.00752)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel, compile_model_for_training
from src.models.registry import ModelRegistry


def _silu(x: tf.Tensor) -> tf.Tensor:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * tf.sigmoid(x)


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class MambaSSMCell(layers.Layer):
    """Selective State Space Model cell.

    At each timestep computes:
      Δ  = softplus(W_Δ · x)              (input-dependent step size)
      B  = W_B · x                         (input-dependent B matrix)
      C  = W_C · x                         (input-dependent C matrix)
      Ā  = exp(Δ ⊗ A)                     (discretised state transition)
      B̄  = Δ ⊗ B                           (discretised input matrix)
      h' = Ā ⊗ h + B̄ ⊗ x                  (state update)
      y  = (C ⊗ h').sum(axis=-1) + D·x    (output)

    All ⊗ are element-wise; A is diagonal (represented as a vector).
    """

    def __init__(self, d_model: int, d_state: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state

    @property
    def state_size(self) -> int:
        # Flattened state: (d_model, d_state)
        return self.d_model * self.d_state

    @property
    def output_size(self) -> int:
        return self.d_model

    def build(self, input_shape: Tuple):
        d, N = self.d_model, self.d_state
        # A: log-parameterised diagonal state matrix, initialised per Mamba paper
        # Precompute with numpy to avoid SymbolicTensor.numpy() issues during tracing
        a_init_np = np.log(np.tile(np.arange(1, N + 1)[np.newaxis, :], [d, 1])).astype(np.float32)
        self.A_log = self.add_weight(
            name="A_log", shape=(d, N), initializer=tf.constant_initializer(a_init_np)
        )
        self.D = self.add_weight(name="D", shape=(d,), initializer="ones")
        # Input-dependent projections
        self.W_delta = self.add_weight(name="W_delta", shape=(d, d), initializer="glorot_uniform")
        self.b_delta = self.add_weight(name="b_delta", shape=(d,), initializer="zeros")
        self.W_B = self.add_weight(name="W_B", shape=(d, N), initializer="glorot_uniform")
        self.W_C = self.add_weight(name="W_C", shape=(d, N), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor, ...]) -> Tuple:
        # inputs: (batch, d_model)
        # states[0]: (batch, d_model * d_state) — flattened h
        batch_size = tf.shape(inputs)[0]
        h = tf.reshape(states[0], (batch_size, self.d_model, self.d_state))

        A = -tf.exp(self.A_log)  # (d_model, d_state) — ensure negative eigenvalues

        # Input-dependent Δ: softplus for positivity
        delta = tf.nn.softplus(inputs @ self.W_delta + self.b_delta)  # (B, d)

        # Discretise A: Ā[b,d,n] = exp(Δ[b,d] * A[d,n])
        A_bar = tf.exp(
            delta[:, :, tf.newaxis] * A[tf.newaxis, :, :]
        )  # (B, d, N)

        # Discretise B: B̄[b,d,n] = Δ[b,d] * (x[b,d] * W_B[d,n])
        B = inputs @ self.W_B  # (B, N) — project input to state space
        B_bar = delta[:, :, tf.newaxis] * (
            inputs[:, :, tf.newaxis] * self.W_B[tf.newaxis, :, :]
        )  # (B, d, N)

        # State update: h' = Ā ⊗ h + B̄
        h_new = A_bar * h + B_bar  # (B, d, N)

        # Output: y[b,d] = sum_n C[b,n] * h'[b,d,n] + D[d] * x[b,d]
        C = inputs @ self.W_C  # (B, N)
        y = tf.reduce_sum(C[:, tf.newaxis, :] * h_new, axis=-1) + self.D * inputs  # (B, d)

        new_state = tf.reshape(h_new, (batch_size, self.d_model * self.d_state))
        return y, [new_state]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "d_state": self.d_state})
        return config


@tf.keras.utils.register_keras_serializable(package="NCMAPSS")
class MambaBlock(layers.Layer):
    """Residual Mamba block with gated SSM and SiLU activation.

    Architecture (per block):
      x_norm = LayerNorm(x)
      [x_ssm, z] = split(Linear(x_norm), 2)   # gate split
      x_conv = Conv1D(x_ssm)                   # local feature mixing
      x_out = SSM(x_conv) * silu(z)            # selective SSM + gating
      y = Linear_out(x_out) + x               # project + residual
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.dropout_rate = dropout
        d_inner = d_model * expand

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.in_proj = layers.Dense(d_inner * 2)           # projects to [x_ssm | z]
        self.conv1d = layers.Conv1D(
            d_inner, kernel_size=3, padding="same", groups=1, activation=None
        )
        self.ssm = layers.RNN(
            MambaSSMCell(d_inner, d_state=d_state), return_sequences=True
        )
        self.out_proj = layers.Dense(d_model)
        self.drop = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        residual = inputs
        x = self.norm(inputs)

        # Split into SSM path and gate path
        xz = self.in_proj(x)           # (B, T, 2*d_inner)
        x_ssm, z = tf.split(xz, 2, axis=-1)  # each (B, T, d_inner)

        # Local convolution + SSM
        x_ssm = self.conv1d(x_ssm)
        x_ssm = self.ssm(x_ssm, training=training)

        # Gated output
        x_out = x_ssm * _silu(z)
        x_out = self.drop(x_out, training=training)
        return self.out_proj(x_out) + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "expand": self.expand,
            "dropout": self.dropout_rate,
        })
        return config


@ModelRegistry.register("mamba_rul")
class MambaRUL(BaseModel):
    """Mamba-inspired Selective SSM for RUL prediction.

    O(N) complexity vs O(N²) for attention-based models. Input-dependent
    state transitions capture long-range degradation dynamics efficiently.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        d_state: int = 16,
        num_layers: int = 4,
        expand: int = 2,
        **kwargs,
    ) -> keras.Model:
        """Build and compile the Mamba RUL model.

        Args:
            input_shape: (sequence_length, num_features)
            units: Model dimension (d_model)
            dense_units: Output dense layer units
            dropout_rate: Dropout probability
            learning_rate: Adam optimizer learning rate
            d_state: SSM state size (N in Mamba paper; default 16)
            num_layers: Number of stacked MambaBlocks (default 4)
            expand: Expansion factor for inner dimension (default 2)

        Returns:
            Compiled Keras model with output shape (batch, 1)
        """
        inputs = layers.Input(shape=input_shape, name="input")

        # Project raw features to model dimension
        x = layers.Dense(units, name="input_proj")(inputs)

        # Stack Mamba blocks
        for i in range(num_layers):
            x = MambaBlock(
                d_model=units,
                d_state=d_state,
                expand=expand,
                dropout=dropout_rate,
                name=f"mamba_block_{i}",
            )(x)

        # Aggregate sequence
        x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
        x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
        x = layers.Dropout(dropout_rate, name="dropout")(x)
        outputs = layers.Dense(1, activation="linear", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="mamba_rul")
        return compile_model_for_training(model, learning_rate=learning_rate)
