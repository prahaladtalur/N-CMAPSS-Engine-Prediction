"""Smoke tests for the STAR two-stage hierarchical transformer."""

import numpy as np
import pytest

from src.models import star_transformer  # noqa: F401 — trigger registration
from src.models.registry import ModelRegistry

SMALL_INPUT = (50, 6)
BATCH = 4


class TestSTARTransformer:
    def test_registers_in_model_registry(self):
        assert "star_transformer" in ModelRegistry.list_models()

    def test_builds_successfully(self):
        model = ModelRegistry.build("star_transformer", input_shape=SMALL_INPUT)
        assert model is not None

    def test_output_shape_is_scalar(self):
        model = ModelRegistry.build("star_transformer", input_shape=SMALL_INPUT)
        assert model.output_shape == (None, 1)

    def test_forward_pass_finite(self):
        model = ModelRegistry.build("star_transformer", input_shape=SMALL_INPUT)
        x = np.random.randn(BATCH, *SMALL_INPUT).astype(np.float32)
        preds = model(x, training=False).numpy()
        assert preds.shape == (BATCH, 1)
        assert np.all(np.isfinite(preds))

    def test_model_is_compiled(self):
        model = ModelRegistry.build("star_transformer", input_shape=SMALL_INPUT)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_gradient_flows(self):
        """Verify training doesn't produce NaN loss."""
        model = ModelRegistry.build(
            "star_transformer", input_shape=SMALL_INPUT, num_layers=1, units=16
        )
        x = np.random.randn(BATCH, *SMALL_INPUT).astype(np.float32)
        y = np.random.rand(BATCH, 1).astype(np.float32) * 100
        model.fit(x, y, epochs=1, verbose=0)
        preds = model(x, training=False).numpy()
        assert np.all(np.isfinite(preds)), "Predictions contain NaN after training"

    def test_different_input_shapes(self):
        """STAR should handle different timestep and feature counts."""
        for shape in [(20, 4), (100, 14), (50, 28)]:
            model = ModelRegistry.build("star_transformer", input_shape=shape, num_layers=1)
            x = np.random.randn(2, *shape).astype(np.float32)
            out = model(x, training=False)
            assert out.shape == (2, 1), f"Wrong output shape for input {shape}: {out.shape}"
