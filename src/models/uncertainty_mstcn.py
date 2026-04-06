"""
Aleatoric uncertainty MSTCN variant.

Identical to the MSTCN architecture up to the shared dense representation,
then splits into two output heads:
  - mean_output : predicted RUL (linear)
  - log_var_output : log-variance of the aleatoric uncertainty (linear)

Concatenated output shape: (batch, 2) — [mean, log_var]
Trained with Negative Log-Likelihood (NLL) loss from base.py.
"""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import BaseModel, nll_loss
from src.models.cata_tcn import ResidualTCNBlock
from src.models.mstcn import GlobalFusionAttention
from src.models.registry import ModelRegistry


@ModelRegistry.register("uncertainty_mstcn")
class UncertaintyMSTCN(BaseModel):
    """MSTCN with an aleatoric uncertainty head.

    Outputs a concatenated (batch, 2) tensor: [mean_rul, log_var].
    Trained with NLL loss; confidence intervals can be derived from log_var.
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 3,
        dilation_rates: List[int] = None,
    ) -> keras.Model:
        """Build and compile the uncertainty MSTCN model.

        Args:
            input_shape: (sequence_length, num_features)
            units: Number of filters in TCN blocks
            dense_units: Number of units in final dense layer
            dropout_rate: Dropout probability
            learning_rate: Optimizer learning rate
            kernel_size: Kernel size for TCN convolutions
            dilation_rates: Dilation rates for multi-scale branches (default: [1, 2, 4, 8])

        Returns:
            Compiled Keras model with output shape (batch, 2): [mean, log_var]
        """
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        num_scales = len(dilation_rates)

        inputs = layers.Input(shape=input_shape, name="input")

        # Multi-Scale TCN: parallel branches with different dilation rates
        tcn_outputs = []
        for i, dilation_rate in enumerate(dilation_rates):
            branch = inputs

            # First TCN block
            branch = ResidualTCNBlock(
                filters=units,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
                name=f"tcn_scale{i}_block1",
            )(branch)

            # Second TCN block (same dilation for deeper receptive field)
            branch = ResidualTCNBlock(
                filters=units,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
                name=f"tcn_scale{i}_block2",
            )(branch)

            tcn_outputs.append(branch)

        # Global Fusion Attention: intelligently combine multi-scale features
        fused = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8, name="global_fusion")(
            tcn_outputs
        )

        # Global pooling to get fixed-size representation
        shared = layers.GlobalAveragePooling1D(name="global_pooling")(fused)

        # Shared dense representation
        shared = layers.Dense(dense_units, activation="relu", name="dense_1")(shared)
        shared = layers.Dropout(dropout_rate, name="dropout")(shared)

        # Dual output heads
        mean_output = layers.Dense(1, activation="linear", name="mean")(shared)
        log_var_output = layers.Dense(1, activation="linear", name="log_var")(shared)
        outputs = layers.Concatenate(name="output")([mean_output, log_var_output])

        model = keras.Model(inputs=inputs, outputs=outputs, name="uncertainty_mstcn")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=nll_loss(),
            metrics=[],  # no standard metrics — output shape is (batch, 2)
        )
        return model
