"""
Smoke tests for all models registered in ModelRegistry.

Verifies that every architecture:
  - Builds without errors for a small input shape
  - Produces scalar output (batch, 1)
  - Is compiled (has an optimizer and loss)
  - Can run one forward pass without crashing

Tests use tiny synthetic data and run on CPU to stay fast.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.models.architectures import ModelRegistry, list_available_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_INPUT_SHAPE = (50, 6)  # (timesteps, features) — minimal, fast


def _build(model_name: str) -> keras.Model:
    return ModelRegistry.build(model_name, input_shape=SMALL_INPUT_SHAPE)


# ---------------------------------------------------------------------------
# Registry metadata tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test the ModelRegistry infrastructure."""

    def test_list_models_returns_nonempty_list(self):
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_known_models_are_registered(self):
        registered = set(list_available_models())
        expected = {
            "lstm",
            "bilstm",
            "gru",
            "bigru",
            "attention_lstm",
            "tcn",
            "cnn_lstm",
            "transformer",
            "resnet_lstm",
            "wavenet",
            "mlp",
            "cnn_gru",
            "inception_lstm",
            "mdfa",
            "cnn_lstm_attention",
            "cata_tcn",
            "ttsnet",
            "atcn",
            "sparse_transformer_bigrcu",
            "mstcn",
        }
        assert expected.issubset(registered), f"Missing models: {expected - registered}"

    def test_build_unknown_model_raises(self):
        with pytest.raises((KeyError, ValueError)):
            ModelRegistry.build("nonexistent_model_xyz", input_shape=SMALL_INPUT_SHAPE)


# ---------------------------------------------------------------------------
# Parametrised smoke tests — one test instance per registered model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", list_available_models())
class TestAllModelsSmoke:
    """Smoke test every registered model."""

    def test_builds_without_error(self, model_name):
        model = _build(model_name)
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_correct_input_shape(self, model_name):
        model = _build(model_name)
        expected = (None, *SMALL_INPUT_SHAPE)
        assert (
            model.input_shape == expected
        ), f"{model_name}: expected input {expected}, got {model.input_shape}"

    def test_scalar_output(self, model_name):
        model = _build(model_name)
        assert model.output_shape == (
            None,
            1,
        ), f"{model_name}: expected output (None, 1), got {model.output_shape}"

    def test_is_compiled(self, model_name):
        model = _build(model_name)
        assert model.optimizer is not None, f"{model_name}: model has no optimizer"
        assert model.loss is not None, f"{model_name}: model has no loss"

    def test_has_mae_metric(self, model_name):
        model = _build(model_name)
        metric_names = [m.name for m in model.metrics]
        assert "mae" in metric_names, f"{model_name}: 'mae' not in metrics {metric_names}"

    def test_forward_pass_finite(self, model_name):
        """One forward pass should produce finite predictions."""
        model = _build(model_name)
        X = np.random.randn(4, *SMALL_INPUT_SHAPE).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (4, 1)
        assert np.all(np.isfinite(preds)), f"{model_name}: non-finite predictions"

    def test_has_nonzero_parameters(self, model_name):
        model = _build(model_name)
        assert model.count_params() > 0, f"{model_name}: model has 0 parameters"
