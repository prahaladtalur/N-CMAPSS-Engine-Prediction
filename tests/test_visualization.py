"""
Tests for training visualization utilities.

Verifies that all plot functions run without errors on synthetic data.
We do not inspect pixel content — only that the functions complete without
raising and (optionally) write files to disk.
"""

import os

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend — no display required in CI
matplotlib.use("Agg")


from src.utils.training_viz import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_model_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(epochs: int = 5, with_val: bool = True):
    """Synthetic training history dict."""
    rng = np.random.default_rng(0)
    h = {
        "loss": list(rng.random(epochs).astype(float)),
        "mae": list(rng.random(epochs).astype(float)),
    }
    if with_val:
        h["val_loss"] = list(rng.random(epochs).astype(float))
        h["val_mae"] = list(rng.random(epochs).astype(float))
    return h


def _make_preds(n: int = 50):
    rng = np.random.default_rng(1)
    y_true = (rng.random(n) * 100).astype(float)
    y_pred = y_true + rng.standard_normal(n) * 5
    return y_true, y_pred


# ---------------------------------------------------------------------------
# plot_training_history
# ---------------------------------------------------------------------------


class TestPlotTrainingHistory:
    def test_runs_without_saving(self):
        """Should complete without raising even with no save_path."""
        plot_training_history(_make_history(), model_name="test_model", save_path=None)

    def test_saves_file_to_disk(self, tmp_path):
        save_path = str(tmp_path / "history.png")
        plot_training_history(_make_history(), model_name="test_model", save_path=save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_works_without_validation_data(self):
        """Should handle histories that have no val_loss/val_mae."""
        plot_training_history(_make_history(with_val=False), model_name="no_val", save_path=None)

    def test_single_epoch(self):
        plot_training_history(_make_history(epochs=1), save_path=None)

    def test_many_epochs(self):
        plot_training_history(_make_history(epochs=100), save_path=None)


# ---------------------------------------------------------------------------
# plot_predictions
# ---------------------------------------------------------------------------


class TestPlotPredictions:
    def test_runs_without_saving(self):
        y_true, y_pred = _make_preds()
        plot_predictions(y_true, y_pred, model_name="test", save_path=None)

    def test_saves_file_to_disk(self, tmp_path):
        y_true, y_pred = _make_preds()
        save_path = str(tmp_path / "preds.png")
        plot_predictions(y_true, y_pred, model_name="test", save_path=save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_single_sample(self):
        plot_predictions(np.array([50.0]), np.array([48.0]), model_name="test", save_path=None)

    def test_perfect_predictions(self):
        y = np.linspace(0, 100, 20)
        plot_predictions(y, y.copy(), model_name="perfect", save_path=None)


# ---------------------------------------------------------------------------
# plot_error_distribution
# ---------------------------------------------------------------------------


class TestPlotErrorDistribution:
    def test_runs_without_saving(self):
        y_true, y_pred = _make_preds()
        plot_error_distribution(y_true, y_pred, model_name="test", save_path=None)

    def test_saves_file_to_disk(self, tmp_path):
        y_true, y_pred = _make_preds()
        save_path = str(tmp_path / "errors.png")
        plot_error_distribution(y_true, y_pred, model_name="test", save_path=save_path)
        assert os.path.exists(save_path)

    def test_zero_errors(self):
        y = np.linspace(1, 100, 30)
        plot_error_distribution(y, y.copy(), model_name="zero_err", save_path=None)


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------


class TestPlotModelComparison:
    def _metrics(self, rmse, r2=0.9):
        return {"rmse": rmse, "mae": rmse * 0.8, "r2": r2, "accuracy_20": 95.0}

    def test_runs_without_saving(self):
        metrics = {
            "model_a": self._metrics(10.0),
            "model_b": self._metrics(8.0, r2=0.92),
        }
        plot_model_comparison(metrics, save_path=None)

    def test_saves_file_to_disk(self, tmp_path):
        metrics = {
            "model_a": self._metrics(10.0),
            "model_b": self._metrics(8.0),
        }
        save_path = str(tmp_path / "comparison.png")
        plot_model_comparison(metrics, save_path=save_path)
        assert os.path.exists(save_path)

    def test_single_model(self):
        plot_model_comparison({"only_model": self._metrics(7.5)}, save_path=None)

    def test_many_models(self):
        metrics = {f"model_{i}": self._metrics(float(i)) for i in range(1, 8)}
        plot_model_comparison(metrics, save_path=None)
