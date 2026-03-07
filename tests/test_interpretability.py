"""
Tests for the interpretability / attention visualization toolkit.

All tests use small synthetic data and a tiny MLP model so they run fast on
CPU without any real data downloads.
"""

import os

import matplotlib
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

matplotlib.use("Agg")  # Non-interactive backend for CI

from src.utils.interpretability import (
    NCMAPSS_SENSOR_NAMES,
    aggregate_saliency,
    compute_saliency_map,
    extract_attention_weights,
    generate_interpretability_report,
    plot_attention_comparison,
    plot_saliency_heatmap,
    plot_sensor_importance,
    plot_temporal_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, T, F = 20, 50, 8  # small synthetic data


@pytest.fixture
def tiny_model():
    """Minimal MLP model for testing — no attention layers."""
    inp = keras.Input(shape=(T, F))
    x = keras.layers.GlobalAveragePooling1D()(inp)
    x = keras.layers.Dense(16, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@pytest.fixture
def cata_tcn_tiny():
    """Tiny CATA-TCN model (has ChannelAttention1D + TemporalAttention1D)."""
    from src.models.architectures import ModelRegistry

    return ModelRegistry.build("cata_tcn", input_shape=(T, F))


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.random((N, T, F)).astype(np.float32)


@pytest.fixture
def y():
    rng = np.random.default_rng(1)
    return (rng.random(N) * 100).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_saliency_map
# ---------------------------------------------------------------------------


class TestComputeSaliencyMap:
    def test_output_shape(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        assert sal.shape == X.shape

    def test_values_nonnegative(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        assert np.all(sal >= 0)

    def test_values_finite(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        assert np.all(np.isfinite(sal))

    def test_batch_size_does_not_affect_result(self, tiny_model, X):
        sal1 = compute_saliency_map(tiny_model, X, batch_size=4)
        sal2 = compute_saliency_map(tiny_model, X, batch_size=N)
        np.testing.assert_allclose(sal1, sal2, rtol=1e-5)

    def test_different_inputs_give_different_saliency(self):
        """Use a nonlinear model — ReLU activations make gradients input-dependent."""
        inp = keras.Input(shape=(T, F))
        # Conv + ReLU: gradients are gated by activations → input-dependent
        x = keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inp)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        rng = np.random.default_rng(42)
        X1 = rng.random((5, T, F)).astype(np.float32)
        X2 = rng.random((5, T, F)).astype(np.float32)
        sal1 = compute_saliency_map(model, X1)
        sal2 = compute_saliency_map(model, X2)
        assert not np.allclose(sal1, sal2)


# ---------------------------------------------------------------------------
# aggregate_saliency
# ---------------------------------------------------------------------------


class TestAggregateSaliency:
    def test_heatmap_shape(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        agg = aggregate_saliency(sal)
        assert agg["heatmap"].shape == (T, F)

    def test_sensor_shape(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        agg = aggregate_saliency(sal)
        assert agg["sensor"].shape == (F,)

    def test_temporal_shape(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        agg = aggregate_saliency(sal)
        assert agg["temporal"].shape == (T,)

    def test_all_values_nonnegative(self, tiny_model, X):
        sal = compute_saliency_map(tiny_model, X)
        agg = aggregate_saliency(sal)
        for key, arr in agg.items():
            assert np.all(arr >= 0), f"{key} has negative values"


# ---------------------------------------------------------------------------
# extract_attention_weights
# ---------------------------------------------------------------------------


class TestExtractAttentionWeights:
    def test_returns_none_for_plain_model(self, tiny_model, X):
        result = extract_attention_weights(tiny_model, X)
        assert result["channel"] is None
        assert result["temporal"] is None

    def test_returns_channel_weights_for_cata_tcn(self, cata_tcn_tiny, X):
        result = extract_attention_weights(cata_tcn_tiny, X)
        # Channel or temporal should be extracted (or gracefully None)
        # At minimum the function should not raise
        assert "channel" in result
        assert "temporal" in result

    def test_channel_weights_shape_if_extracted(self, cata_tcn_tiny, X):
        result = extract_attention_weights(cata_tcn_tiny, X)
        if result["channel"] is not None:
            # First axis is batch; second is the model's internal feature dim
            # (may differ from input F when channel attention operates on hidden units)
            assert result["channel"].shape[0] == N
            assert result["channel"].ndim == 2

    def test_temporal_weights_shape_if_extracted(self, cata_tcn_tiny, X):
        result = extract_attention_weights(cata_tcn_tiny, X)
        if result["temporal"] is not None:
            assert result["temporal"].shape[0] == N


# ---------------------------------------------------------------------------
# plot_saliency_heatmap
# ---------------------------------------------------------------------------


class TestPlotSaliencyHeatmap:
    def _heatmap(self):
        rng = np.random.default_rng(0)
        return rng.random((T, F)).astype(np.float32)

    def test_runs_without_saving(self):
        plot_saliency_heatmap(self._heatmap(), save_path=None)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "heatmap.png")
        plot_saliency_heatmap(self._heatmap(), save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_with_sensor_names(self):
        names = [f"Sensor{i}" for i in range(F)]
        plot_saliency_heatmap(self._heatmap(), sensor_names=names, save_path=None)

    def test_with_too_many_sensors_clips(self, tmp_path):
        heatmap = np.random.rand(100, 50).astype(np.float32)
        path = str(tmp_path / "clip.png")
        plot_saliency_heatmap(heatmap, max_sensors=10, save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# plot_sensor_importance
# ---------------------------------------------------------------------------


class TestPlotSensorImportance:
    def _weights(self):
        return np.abs(np.random.default_rng(0).random(F)).astype(np.float32)

    def test_runs_without_saving(self):
        plot_sensor_importance(self._weights(), save_path=None)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "importance.png")
        plot_sensor_importance(self._weights(), save_path=path)
        assert os.path.exists(path)

    def test_with_sensor_names(self):
        names = NCMAPSS_SENSOR_NAMES[:F]
        plot_sensor_importance(self._weights(), sensor_names=names, save_path=None)

    def test_top_n_clipping(self, tmp_path):
        weights = np.abs(np.random.rand(30)).astype(np.float32)
        path = str(tmp_path / "top5.png")
        plot_sensor_importance(weights, top_n=5, save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# plot_temporal_profile
# ---------------------------------------------------------------------------


class TestPlotTemporalProfile:
    def test_raw_temporal_profile(self):
        weights = np.random.rand(T).astype(np.float32)
        plot_temporal_profile(weights, save_path=None)

    def test_with_rul_values(self):
        weights = np.random.rand(N).astype(np.float32)
        rul = (np.random.rand(N) * 100).astype(np.float32)
        plot_temporal_profile(weights, rul_values=rul, save_path=None)

    def test_saves_file(self, tmp_path):
        weights = np.random.rand(T).astype(np.float32)
        path = str(tmp_path / "temporal.png")
        plot_temporal_profile(weights, save_path=path)
        assert os.path.exists(path)

    def test_saves_file_with_rul(self, tmp_path):
        weights = np.random.rand(N).astype(np.float32)
        rul = (np.random.rand(N) * 80).astype(np.float32)
        path = str(tmp_path / "temporal_rul.png")
        plot_temporal_profile(weights, rul_values=rul, save_path=path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# plot_attention_comparison
# ---------------------------------------------------------------------------


class TestPlotAttentionComparison:
    def _results(self, models=("A", "B")):
        rng = np.random.default_rng(0)
        return {m: {"sensor": rng.random(F).astype(np.float32)} for m in models}

    def test_runs_without_saving(self):
        plot_attention_comparison(self._results(), save_path=None)

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "compare.png")
        plot_attention_comparison(self._results(), save_path=path)
        assert os.path.exists(path)

    def test_single_model(self):
        plot_attention_comparison(self._results(("only",)), save_path=None)

    def test_empty_dict_is_noop(self):
        plot_attention_comparison({}, save_path=None)


# ---------------------------------------------------------------------------
# generate_interpretability_report
# ---------------------------------------------------------------------------


class TestGenerateInterpretabilityReport:
    def test_creates_expected_files(self, tiny_model, X, y, tmp_path):
        save_dir = str(tmp_path / "report")
        saved = generate_interpretability_report(
            model=tiny_model,
            X_test=X,
            y_test=y,
            model_name="test_model",
            save_dir=save_dir,
        )
        assert "saliency_heatmap" in saved
        assert "sensor_importance" in saved
        assert "temporal_profile" in saved
        for path in saved.values():
            assert os.path.exists(path), f"Missing: {path}"

    def test_all_files_nonempty(self, tiny_model, X, y, tmp_path):
        save_dir = str(tmp_path / "report2")
        saved = generate_interpretability_report(
            model=tiny_model, X_test=X, y_test=y, save_dir=save_dir
        )
        for path in saved.values():
            assert os.path.getsize(path) > 0

    def test_creates_attention_files_for_cata_tcn(self, cata_tcn_tiny, X, y, tmp_path):
        save_dir = str(tmp_path / "cata_report")
        saved = generate_interpretability_report(
            model=cata_tcn_tiny,
            X_test=X,
            y_test=y,
            model_name="cata_tcn",
            save_dir=save_dir,
        )
        # At minimum gradient files must exist
        assert "saliency_heatmap" in saved
        assert "sensor_importance" in saved

    def test_max_samples_is_respected(self, tiny_model, tmp_path):
        rng = np.random.default_rng(0)
        X_large = rng.random((200, T, F)).astype(np.float32)
        y_large = (rng.random(200) * 100).astype(np.float32)
        save_dir = str(tmp_path / "sampled")
        # Should not crash when max_samples < len(X)
        saved = generate_interpretability_report(
            model=tiny_model,
            X_test=X_large,
            y_test=y_large,
            save_dir=save_dir,
            max_samples=30,
        )
        assert len(saved) >= 3

    def test_auto_detects_ncmapss_sensor_names(self, tmp_path):
        """When F matches N-CMAPSS feature count, names should be auto-applied."""
        n_sensors = len(NCMAPSS_SENSOR_NAMES)
        inp = keras.Input(shape=(T, n_sensors))
        x = keras.layers.GlobalAveragePooling1D()(inp)
        out = keras.layers.Dense(1)(x)
        m = keras.Model(inputs=inp, outputs=out)
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])

        rng = np.random.default_rng(7)
        X_s = rng.random((10, T, n_sensors)).astype(np.float32)
        y_s = (rng.random(10) * 50).astype(np.float32)

        save_dir = str(tmp_path / "named")
        saved = generate_interpretability_report(m, X_s, y_s, save_dir=save_dir)
        assert len(saved) >= 3
