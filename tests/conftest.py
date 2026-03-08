"""
Pytest configuration and shared fixtures for test suite.

Provides synthetic data, model configurations, and utilities for testing
without requiring real data downloads or external dependencies.
"""

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def small_input_shape():
    """Small input shape for fast model testing."""
    return (10, 5)  # 10 timesteps, 5 features


@pytest.fixture
def synthetic_data(small_input_shape):
    """
    Generate small synthetic dataset for testing.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val) with small random data
    """
    timesteps, features = small_input_shape
    n_train = 20
    n_val = 10

    # Random sensor data
    X_train = np.random.randn(n_train, timesteps, features).astype(np.float32)
    X_val = np.random.randn(n_val, timesteps, features).astype(np.float32)

    # Random RUL targets (0-100 cycles)
    y_train = np.random.uniform(0, 100, size=n_train).astype(np.float32)
    y_val = np.random.uniform(0, 100, size=n_val).astype(np.float32)

    return X_train, y_train, X_val, y_val


@pytest.fixture
def default_model_params():
    """Default hyperparameters for model building."""
    return {
        "units": 16,  # Small for fast tests
        "dense_units": 8,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
    }


@pytest.fixture
def all_model_names():
    """List of all registered model names to test."""
    return [
        # RNN models
        "lstm",
        "bilstm",
        "gru",
        "bigru",
        # Attention models
        "attention_lstm",
        "transformer",
        "resnet_lstm",
        # CNN models
        "tcn",
        "wavenet",
        # Hybrid models
        "cnn_lstm",
        "cnn_gru",
        "inception_lstm",
        # Baseline
        "mlp",
    ]


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def disable_gpu():
    """Disable GPU for consistent test behavior and faster execution."""
    tf.config.set_visible_devices([], 'GPU')
