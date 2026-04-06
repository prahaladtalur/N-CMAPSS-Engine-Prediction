"""Tests for aleatoric uncertainty MSTCN and NLL loss."""

import numpy as np
import pytest
import tensorflow as tf

from src.models import uncertainty_mstcn  # noqa: F401 — trigger registration
from src.models.base import nll_loss
from src.models.registry import ModelRegistry


SMALL_INPUT_SHAPE = (50, 6)
BATCH_SIZE = 4


class TestNLLLoss:
    """Tests for the NLL loss function."""

    def test_nll_loss_returns_scalar(self):
        loss_fn = nll_loss()
        y_true = tf.constant([[10.0], [20.0], [30.0]])
        # Output: [mean, log_var] — low log_var, mean close to y_true
        y_pred = tf.constant([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
        loss_val = loss_fn(y_true, y_pred)
        assert loss_val.shape == ()

    def test_nll_loss_lower_when_mean_closer(self):
        """Loss should be lower when mean is closer to target."""
        loss_fn = nll_loss()
        y_true = tf.constant([[10.0]])
        good_pred = tf.constant([[10.0, 0.0]])  # exact mean, log_var=0
        bad_pred = tf.constant([[50.0, 0.0]])  # far from target
        assert loss_fn(y_true, good_pred) < loss_fn(y_true, bad_pred)

    def test_nll_loss_finite_on_extreme_log_var(self):
        """Loss should remain finite when log_var is clamped."""
        loss_fn = nll_loss()
        y_true = tf.constant([[10.0]])
        y_pred = tf.constant([[10.0, 100.0]])  # extreme log_var — clamped to 10
        loss_val = loss_fn(y_true, y_pred)
        assert tf.math.is_finite(loss_val)


class TestUncertaintyMSTCN:
    """Tests for the uncertainty_mstcn model."""

    def test_builds_successfully(self):
        model = ModelRegistry.build("uncertainty_mstcn", input_shape=SMALL_INPUT_SHAPE)
        assert model is not None

    def test_output_shape_is_batch_2(self):
        model = ModelRegistry.build("uncertainty_mstcn", input_shape=SMALL_INPUT_SHAPE)
        assert model.output_shape == (None, 2), f"Expected (None, 2), got {model.output_shape}"

    def test_forward_pass_produces_finite_values(self):
        model = ModelRegistry.build("uncertainty_mstcn", input_shape=SMALL_INPUT_SHAPE)
        x = np.random.randn(BATCH_SIZE, *SMALL_INPUT_SHAPE).astype(np.float32)
        preds = model(x, training=False)
        assert preds.shape == (BATCH_SIZE, 2)
        assert np.all(np.isfinite(preds.numpy())), "Predictions contain non-finite values"

    def test_std_from_log_var_is_positive(self):
        """Std derived from log_var output should be > 0."""
        model = ModelRegistry.build("uncertainty_mstcn", input_shape=SMALL_INPUT_SHAPE)
        x = np.random.randn(BATCH_SIZE, *SMALL_INPUT_SHAPE).astype(np.float32)
        preds = model(x, training=False).numpy()
        log_var = preds[:, 1]
        std = np.sqrt(np.exp(np.clip(log_var, -10, 10)))
        assert np.all(std > 0), "All std values should be positive"

    def test_is_registered_in_model_registry(self):
        from src.models.registry import ModelRegistry as MR

        assert "uncertainty_mstcn" in MR.list_models()

    def test_model_has_optimizer_and_loss(self):
        model = ModelRegistry.build("uncertainty_mstcn", input_shape=SMALL_INPUT_SHAPE)
        assert model.optimizer is not None
        assert model.loss is not None
