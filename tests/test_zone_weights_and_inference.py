"""Tests for critical-zone sample weighting, MC Dropout, and monotonic inference."""

import numpy as np
import pytest


class TestComputeZoneWeights:
    """Tests for compute_zone_weights() in train_model.py."""

    def _fn(self):
        from train_model import compute_zone_weights

        return compute_zone_weights

    def test_returns_same_shape(self):
        fn = self._fn()
        y = np.array([5.0, 35.0, 100.0])
        w = fn(y)
        assert w.shape == y.shape

    def test_critical_zone_weight(self):
        fn = self._fn()
        y = np.array([0.0, 15.0, 30.0])
        w = fn(y)
        assert np.allclose(w, 2.5), f"Expected 2.5 for critical zone, got {w}"

    def test_warning_zone_weight(self):
        fn = self._fn()
        y = np.array([31.0, 50.0, 80.0])
        w = fn(y)
        assert np.allclose(w, 1.5), f"Expected 1.5 for warning zone, got {w}"

    def test_normal_zone_weight(self):
        fn = self._fn()
        y = np.array([81.0, 120.0, 200.0])
        w = fn(y)
        assert np.allclose(w, 1.0), f"Expected 1.0 for normal zone, got {w}"

    def test_dtype_is_float32(self):
        fn = self._fn()
        y = np.array([10.0, 50.0, 100.0])
        w = fn(y)
        assert w.dtype == np.float32


class TestMCDropoutUncertainty:
    """Tests for RULPredictor.predict_with_uncertainty()."""

    def _make_predictor(self):
        """Build a minimal RULPredictor with a tiny MSTCN model."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Simple model with dropout (required for MC Dropout to work)
        inputs = layers.Input(shape=(20, 4))
        x = layers.Dense(16, activation="relu")(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")

        from predict import RULPredictor

        predictor = RULPredictor.__new__(RULPredictor)
        predictor.model_specs = [
            {
                "model": model,
                "max_seq_length": 20,
                "scaler": None,
                "target_transform": {"method": "none", "scale_min": 0.0, "scale_max": 1.0},
                "name": "test_model",
            }
        ]
        predictor.ensemble = None
        predictor.ensemble_weights = None
        return predictor

    def test_returns_tuple_of_two_floats(self):
        predictor = self._make_predictor()
        sequence = np.random.randn(20, 4).astype(np.float32)
        mean, std = predictor.predict_with_uncertainty(sequence, n_passes=10)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_std_is_non_negative(self):
        predictor = self._make_predictor()
        sequence = np.random.randn(20, 4).astype(np.float32)
        _, std = predictor.predict_with_uncertainty(sequence, n_passes=10)
        assert std >= 0.0

    def test_mean_is_finite(self):
        predictor = self._make_predictor()
        sequence = np.random.randn(20, 4).astype(np.float32)
        mean, _ = predictor.predict_with_uncertainty(sequence, n_passes=10)
        assert np.isfinite(mean)


class TestMonotonicPredictions:
    """Tests for RULPredictor.predict_monotonic()."""

    def _make_predictor_with_known_predictions(self, predictions):
        """Build a mock predictor that returns given predictions in sequence."""
        from unittest.mock import MagicMock

        from predict import RULPredictor

        predictor = RULPredictor.__new__(RULPredictor)
        predictor.model_specs = [{}]  # non-empty
        predictor.ensemble = None
        predictor.ensemble_weights = None

        # Patch predict_single to return values from the list
        call_count = [0]

        def mock_predict_single(seq):
            idx = call_count[0]
            call_count[0] += 1
            return {"prediction": predictions[idx]}

        predictor.predict_single = mock_predict_single
        return predictor

    def test_output_is_non_increasing(self):
        preds = [100.0, 95.0, 105.0, 80.0, 85.0]  # non-monotonic raw preds
        predictor = self._make_predictor_with_known_predictions(preds)
        sequences = [np.zeros((5, 3)) for _ in preds]
        result = predictor.predict_monotonic(sequences)
        # After cumulative min: [100, 95, 95, 80, 80]
        for i in range(1, len(result)):
            assert result[i] <= result[i - 1], f"result[{i}]={result[i]} > result[{i-1}]={result[i-1]}"

    def test_already_monotonic_is_unchanged(self):
        preds = [100.0, 90.0, 80.0, 70.0]
        predictor = self._make_predictor_with_known_predictions(preds)
        sequences = [np.zeros((5, 3)) for _ in preds]
        result = predictor.predict_monotonic(sequences)
        assert np.allclose(result, preds)

    def test_output_shape(self):
        preds = [50.0, 40.0, 45.0]
        predictor = self._make_predictor_with_known_predictions(preds)
        sequences = [np.zeros((5, 3)) for _ in preds]
        result = predictor.predict_monotonic(sequences)
        assert result.shape == (3,)
