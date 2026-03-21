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


class TestJsonSafety:
    """Test JSON-safe serialization helpers used by training outputs."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from train_model import make_json_safe

        self.make_json_safe = make_json_safe

    def test_make_json_safe_converts_nested_numpy_scalars(self, tmp_path):
        payload = {
            "metrics_mean": {"rmse_normalized": np.float32(0.1107)},
            "per_seed": {
                "42": {"rmse_normalized": np.float32(0.1124), "best_epoch": np.int64(13)}
            },
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
