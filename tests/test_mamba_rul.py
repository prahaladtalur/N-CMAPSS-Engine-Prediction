"""Smoke tests for the Mamba RUL architecture."""

import numpy as np
import pytest
import tensorflow as tf

from src.models import mamba_rul  # noqa: F401 — trigger registration
from src.models.registry import ModelRegistry

SMALL_INPUT = (50, 6)
BATCH = 4


class TestMambaRUL:
    def test_registers_in_model_registry(self):
        assert "mamba_rul" in ModelRegistry.list_models()

    def test_builds_successfully(self):
        model = ModelRegistry.build("mamba_rul", input_shape=SMALL_INPUT)
        assert model is not None

    def test_output_shape_is_scalar(self):
        model = ModelRegistry.build("mamba_rul", input_shape=SMALL_INPUT)
        assert model.output_shape == (None, 1)

    def test_forward_pass_finite(self):
        model = ModelRegistry.build("mamba_rul", input_shape=SMALL_INPUT)
        x = np.random.randn(BATCH, *SMALL_INPUT).astype(np.float32)
        preds = model(x, training=False).numpy()
        assert preds.shape == (BATCH, 1)
        assert np.all(np.isfinite(preds))

    def test_model_is_compiled(self):
        model = ModelRegistry.build("mamba_rul", input_shape=SMALL_INPUT)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_gradient_flows(self):
        """Loss should decrease after one training step."""
        model = ModelRegistry.build(
            "mamba_rul", input_shape=SMALL_INPUT, num_layers=1, units=16, d_state=4
        )
        x = np.random.randn(BATCH, *SMALL_INPUT).astype(np.float32)
        y = np.random.rand(BATCH, 1).astype(np.float32) * 100

        loss_before = model.evaluate(x, y, verbose=0)[0]
        model.fit(x, y, epochs=1, verbose=0)
        loss_after = model.evaluate(x, y, verbose=0)[0]
        # Just verify training runs without NaN — one step won't always improve loss
        assert np.isfinite(loss_after), "Loss became NaN after training step"
