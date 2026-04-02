"""
Tests for training pipeline helper functions.

Covers prepare_sequences and normalize_data without requiring real data
downloads, GPU, or W&B connectivity.

Note: tests for set_seeds() and get_git_hash() live alongside those functions
(added as part of issue #22 / feat/reproducibility-seeds-22).
"""

import json
import numpy as np
import pytest

# Import the pure helper functions directly from train_model.
# We do this lazily inside tests where needed to avoid importing heavy
# top-level dependencies (wandb, tensorflow) before they are ready.


# ---------------------------------------------------------------------------
# prepare_sequences
# ---------------------------------------------------------------------------


class TestPrepareSequences:
    """Test the sequence flattening helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import prepare_sequences

        self.prepare_sequences = prepare_sequences

    def test_basic_shape(self, seq_list_X, seq_list_y):
        X, y = self.prepare_sequences(seq_list_X, seq_list_y)
        # 3 units × 4 cycles = 12 samples
        assert X.shape == (12, 100, 8)
        assert y.shape == (12,)

    def test_max_sequence_length_truncates(self, seq_list_X, seq_list_y):
        X, y = self.prepare_sequences(seq_list_X, seq_list_y, max_sequence_length=40)
        # timesteps clipped to 40
        assert X.shape == (12, 40, 8)
        assert y.shape == (12,)

    def test_max_seq_length_larger_than_input_is_noop(self, seq_list_X, seq_list_y):
        X, y = self.prepare_sequences(seq_list_X, seq_list_y, max_sequence_length=200)
        assert X.shape == (12, 100, 8)

    def test_takes_last_n_timesteps(self, seq_list_X, seq_list_y):
        """When truncating, the LAST timesteps should be kept."""
        # Construct sequences where the final timestep value is distinctive
        unit_data = np.zeros((1, 100, 4), dtype=np.float32)
        unit_data[0, -10:, :] = 99.0  # last 10 rows are 99
        X_in = [unit_data]
        y_in = [np.array([5.0], dtype=np.float32)]

        X, _ = self.prepare_sequences(X_in, y_in, max_sequence_length=10)
        assert X.shape == (1, 10, 4)
        assert np.all(X[0] == 99.0)

    def test_empty_input_raises_or_returns_empty(self):
        X, y = self.prepare_sequences([], [])
        # Depending on implementation, shape could be (0,) or raise
        assert len(X) == 0
        assert len(y) == 0

    def test_output_types(self, seq_list_X, seq_list_y):
        X, y = self.prepare_sequences(seq_list_X, seq_list_y)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------


class TestNormalizeData:
    """Test StandardScaler-based normalization helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import normalize_data

        self.normalize_data = normalize_data

    def _make_data(self, n=20, t=30, f=4, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, t, f)).astype(np.float32)

    def test_train_becomes_zero_mean(self):
        X_train = self._make_data()
        X_norm, _, _, _ = self.normalize_data(X_train)
        # Mean across samples should be ~0 per feature
        flat = X_norm.reshape(-1, X_norm.shape[-1])
        assert np.allclose(flat.mean(axis=0), 0, atol=1e-5)

    def test_train_becomes_unit_variance(self):
        X_train = self._make_data(n=200)
        X_norm, _, _, _ = self.normalize_data(X_train)
        flat = X_norm.reshape(-1, X_norm.shape[-1])
        assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)

    def test_val_and_test_transformed_but_not_fitted(self):
        X_train = self._make_data(seed=0)
        X_val = self._make_data(seed=1)
        X_test = self._make_data(seed=2)

        X_train_n, X_val_n, X_test_n, scaler = self.normalize_data(X_train, X_val, X_test)

        assert X_val_n is not None
        assert X_test_n is not None
        assert X_val_n.shape == X_val.shape
        assert X_test_n.shape == X_test.shape

    def test_scaler_is_returned(self):
        X_train = self._make_data()
        _, _, _, scaler = self.normalize_data(X_train)
        assert scaler is not None
        # Scaler should be fitted (has mean_ attribute)
        assert hasattr(scaler, "mean_")

    def test_shape_preserved(self):
        X_train = self._make_data(n=10, t=50, f=6)
        X_norm, _, _, _ = self.normalize_data(X_train)
        assert X_norm.shape == X_train.shape

    def test_no_val_test_returns_none(self):
        X_train = self._make_data()
        _, X_val_n, X_test_n, _ = self.normalize_data(X_train)
        assert X_val_n is None
        assert X_test_n is None


# ---------------------------------------------------------------------------
# target preprocessing helpers
# ---------------------------------------------------------------------------


class TestTargetTransforms:
    """Test issue-25 target preprocessing helpers."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import clip_rul_targets, inverse_transform_targets, transform_targets

        self.clip_rul_targets = clip_rul_targets
        self.inverse_transform_targets = inverse_transform_targets
        self.transform_targets = transform_targets

    def test_clip_rul_targets_caps_values(self):
        y = np.array([10.0, 80.0, 150.0], dtype=np.float32)
        clipped = self.clip_rul_targets(y, max_rul=100.0)
        assert np.array_equal(clipped, np.array([10.0, 80.0, 100.0], dtype=np.float32))

    def test_transform_targets_none_preserves_metric_space(self):
        y_train = np.array([20.0, 50.0, 140.0], dtype=np.float32)
        y_train_t, y_val_t, y_test_t, meta = self.transform_targets(
            y_train,
            np.array([130.0], dtype=np.float32),
            np.array([160.0], dtype=np.float32),
            clip_value=125.0,
            scaling="none",
        )
        assert np.array_equal(y_train_t, np.array([20.0, 50.0, 125.0], dtype=np.float32))
        assert np.array_equal(y_val_t, np.array([125.0], dtype=np.float32))
        assert np.array_equal(y_test_t, np.array([125.0], dtype=np.float32))
        assert meta["scale_min"] == 20.0
        assert meta["scale_max"] == 125.0

    def test_transform_targets_minmax_is_reversible(self):
        y_train = np.array([25.0, 75.0, 125.0], dtype=np.float32)
        y_train_t, y_val_t, y_test_t, meta = self.transform_targets(
            y_train,
            np.array([50.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            scaling="minmax",
        )
        assert np.allclose(y_train_t, np.array([0.0, 0.5, 1.0], dtype=np.float32))
        assert np.allclose(y_val_t, np.array([0.25], dtype=np.float32))
        assert np.allclose(y_test_t, np.array([0.75], dtype=np.float32))
        restored = self.inverse_transform_targets(y_test_t, meta)
        assert np.allclose(restored, np.array([100.0], dtype=np.float32))

    def test_transform_targets_rejects_unknown_scaling(self):
        with pytest.raises(ValueError, match="Unsupported target scaling"):
            self.transform_targets(np.array([1.0], dtype=np.float32), scaling="zscore")


class TestOperatingConditionResidualizer:
    @pytest.fixture(autouse=True)
    def _import(self):
        from src.data.preprocessing import OperatingConditionResidualizer

        self.OperatingConditionResidualizer = OperatingConditionResidualizer

    def test_residualizer_removes_linear_operating_condition_signal(self):
        operating = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
                [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]],
            ],
            dtype=np.float32,
        )
        sensor = 2.0 * operating[..., 0] - 0.5 * operating[..., 1] + 7.0
        unit = np.concatenate([operating, sensor[..., np.newaxis]], axis=-1)

        residualizer = self.OperatingConditionResidualizer().fit([unit])
        transformed = residualizer.transform(unit)

        assert np.allclose(transformed[..., 4], 0.0, atol=1e-5)
        assert np.allclose(transformed[..., :4], operating)

    def test_residualizer_models_time_position_within_cycle(self):
        operating = np.ones((2, 3, 4), dtype=np.float32)
        time_position = np.linspace(0.0, 1.0, num=3, dtype=np.float32)
        sensor = np.broadcast_to(5.0 * time_position[None, :] + 2.0, (2, 3))
        unit = np.concatenate([operating, sensor[..., np.newaxis]], axis=-1)

        residualizer = self.OperatingConditionResidualizer().fit([unit])
        transformed = residualizer.transform(unit)

        assert np.allclose(transformed[..., 4], 0.0, atol=1e-5)

    def test_residualizer_fits_only_healthy_cycles_when_labels_provided(self):
        operating = np.ones((2, 3, 4), dtype=np.float32)
        healthy_sensor = np.full((3,), 10.0, dtype=np.float32)
        degraded_sensor = np.full((3,), 25.0, dtype=np.float32)
        unit = np.concatenate(
            [
                operating,
                np.stack([healthy_sensor, degraded_sensor], axis=0)[..., np.newaxis],
            ],
            axis=-1,
        )
        labels = [np.array([65.0, 10.0], dtype=np.float32)]

        residualizer = self.OperatingConditionResidualizer().fit([unit], labels=labels)
        transformed = residualizer.transform(unit)

        assert np.allclose(transformed[0, ..., 4], 0.0, atol=1e-5)
        assert np.allclose(transformed[1, ..., 4], 15.0, atol=1e-5)


class TestJsonSafety:
    """Test JSON-safe serialization helpers used by training outputs."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import make_json_safe

        self.make_json_safe = make_json_safe

    def test_make_json_safe_converts_nested_numpy_scalars(self, tmp_path):
        payload = {
            "metrics_mean": {"rmse_normalized": np.float32(0.1107)},
            "per_seed": {"42": {"rmse_normalized": np.float32(0.1124), "best_epoch": np.int64(13)}},
        }

        safe_payload = self.make_json_safe(payload)
        output_path = tmp_path / "multiseed_summary.json"
        with open(output_path, "w") as f:
            json.dump(safe_payload, f, indent=2)

        with open(output_path) as f:
            written = json.load(f)

        assert written["metrics_mean"]["rmse_normalized"] == pytest.approx(0.1107)
        assert written["per_seed"]["42"]["rmse_normalized"] == pytest.approx(0.1124)
        assert written["per_seed"]["42"]["best_epoch"] == 13


class TestTrainingLoggingHelpers:
    """Test pure helpers used to build W&B logging payloads."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import (
            build_best_epoch_summary,
            build_dataset_manifest,
            build_final_training_metrics,
        )

        self.build_best_epoch_summary = build_best_epoch_summary
        self.build_dataset_manifest = build_dataset_manifest
        self.build_final_training_metrics = build_final_training_metrics

    def test_build_final_training_metrics_includes_rmse(self):
        history = {
            "loss": [5.0, 4.0],
            "rmse": [2.2, 2.0],
            "mae": [1.8, 1.6],
            "val_loss": [4.8, 3.9],
            "val_rmse": [2.1, 1.9],
        }

        metrics = self.build_final_training_metrics(history)

        assert metrics["epochs_trained"] == 2
        assert metrics["final_train_loss"] == pytest.approx(4.0)
        assert metrics["final_train_rmse"] == pytest.approx(2.0)
        assert metrics["final_val_rmse"] == pytest.approx(1.9)

    def test_build_best_epoch_summary_prefers_val_loss(self):
        history = {
            "loss": [5.0, 4.6, 4.4],
            "rmse": [2.4, 2.2, 2.1],
            "val_loss": [4.8, 4.1, 4.3],
            "val_rmse": [2.3, 2.0, 2.1],
        }

        summary = self.build_best_epoch_summary(history)

        assert summary["training/best_epoch"] == 2
        assert summary["training/best_monitor_name"] == "val_loss"
        assert summary["training/best_val_rmse"] == pytest.approx(2.0)
        assert summary["training/best_rmse"] == pytest.approx(2.2)

    def test_build_dataset_manifest_tracks_local_cache_only(self, monkeypatch, tmp_path):
        data_root = tmp_path / "data-root"
        data_root.mkdir()
        cached_file = data_root / "cache.bin"
        cached_file.write_bytes(b"123456")
        monkeypatch.setenv("RUL_DATASETS_DATA_ROOT", str(data_root))

        unit_X = np.ones((2, 5, 3), dtype=np.float32)
        unit_y = np.array([10.0, 5.0], dtype=np.float32)

        manifest = self.build_dataset_manifest(
            dev_X=[unit_X],
            dev_y=[unit_y],
            val_X=None,
            val_y=None,
            test_X=[unit_X],
            test_y=[unit_y],
            config={"fd": 3},
        )

        assert manifest["fd"] == 3
        assert manifest["cache"]["exists"] is True
        assert manifest["cache"]["file_count"] == 1
        assert manifest["cache"]["size_bytes"] == 6
        assert manifest["raw_data_uploaded_to_wandb"] is False
        assert manifest["splits"]["dev"]["cycles"] == 2
        assert manifest["splits"]["test"]["num_features"] == 3
