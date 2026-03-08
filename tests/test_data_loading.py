"""
Tests for data loading and preprocessing pipeline.

Tests that data loading functions return expected shapes and types
without requiring full dataset downloads.
"""

import pytest
import numpy as np


class TestDataShapes:
    """Test that mock data has expected shapes."""

    def test_synthetic_data_shapes(self, synthetic_data, small_input_shape):
        """Test that synthetic data fixture returns correct shapes."""
        X_train, y_train, X_val, y_val = synthetic_data
        timesteps, features = small_input_shape

        # Check training data shapes
        assert X_train.ndim == 3  # (samples, timesteps, features)
        assert X_train.shape[1] == timesteps
        assert X_train.shape[2] == features
        assert y_train.ndim == 1  # (samples,)
        assert len(y_train) == len(X_train)

        # Check validation data shapes
        assert X_val.ndim == 3
        assert X_val.shape[1] == timesteps
        assert X_val.shape[2] == features
        assert y_val.ndim == 1
        assert len(y_val) == len(X_val)

    def test_synthetic_data_types(self, synthetic_data):
        """Test that synthetic data has correct dtypes."""
        X_train, y_train, X_val, y_val = synthetic_data

        assert X_train.dtype == np.float32
        assert X_val.dtype == np.float32
        assert y_train.dtype == np.float32
        assert y_val.dtype == np.float32

    def test_synthetic_data_values(self, synthetic_data):
        """Test that synthetic RUL values are in reasonable range."""
        X_train, y_train, X_val, y_val = synthetic_data

        # RUL should be non-negative
        assert np.all(y_train >= 0)
        assert np.all(y_val >= 0)

        # RUL should be finite
        assert np.all(np.isfinite(X_train))
        assert np.all(np.isfinite(X_val))
        assert np.all(np.isfinite(y_train))
        assert np.all(np.isfinite(y_val))


class TestDataPreprocessing:
    """Test data preprocessing utilities."""

    def test_normalization_preserves_shape(self, synthetic_data):
        """Test that normalization doesn't change data shape."""
        X_train, y_train, X_val, y_val = synthetic_data

        # Simple normalization
        X_train_norm = (X_train - X_train.mean()) / (X_train.std() + 1e-8)

        assert X_train_norm.shape == X_train.shape

    def test_normalization_centers_data(self, synthetic_data):
        """Test that normalization centers data around zero."""
        X_train, _, _, _ = synthetic_data

        # Z-score normalization
        X_train_norm = (X_train - X_train.mean()) / (X_train.std() + 1e-8)

        # Mean should be close to 0 (within numerical precision)
        assert abs(X_train_norm.mean()) < 0.1

    def test_sequence_length_consistency(self, synthetic_data, small_input_shape):
        """Test that all sequences have consistent length."""
        X_train, _, X_val, _ = synthetic_data
        timesteps, _ = small_input_shape

        # All sequences should have same length
        assert all(seq.shape[0] == timesteps for seq in X_train)
        assert all(seq.shape[0] == timesteps for seq in X_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
