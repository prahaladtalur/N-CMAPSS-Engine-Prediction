"""
Shared pytest fixtures for the N-CMAPSS test suite.

All fixtures use small synthetic data so tests run fast on CPU without any
real dataset downloads or GPU.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

# Small enough to be fast; large enough that TCN/dilated layers don't complain.
SMALL_INPUT_SHAPE = (100, 8)  # (timesteps, features)
SMALL_BATCH = 16


@pytest.fixture
def input_shape():
    """Return a small (timesteps, features) tuple."""
    return SMALL_INPUT_SHAPE


@pytest.fixture
def tiny_input_shape():
    """Minimal shape for very fast builds (used in registry smoke tests)."""
    return (50, 6)


@pytest.fixture
def X_batch(input_shape):
    """Random float32 batch: (SMALL_BATCH, timesteps, features)."""
    rng = np.random.default_rng(0)
    return rng.random((SMALL_BATCH, *input_shape)).astype(np.float32)


@pytest.fixture
def y_batch():
    """Random float32 RUL labels in [0, 100]."""
    rng = np.random.default_rng(0)
    return (rng.random(SMALL_BATCH) * 100).astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence-list fixtures (as returned by get_datasets)
# ---------------------------------------------------------------------------


@pytest.fixture
def seq_list_X():
    """Mimic get_datasets output: list of per-unit arrays (num_cycles, T, F)."""
    rng = np.random.default_rng(1)
    # 3 units, each with 4 cycles of length 100 and 8 sensors
    return [rng.random((4, 100, 8)).astype(np.float32) for _ in range(3)]


@pytest.fixture
def seq_list_y():
    """Matching RUL labels for seq_list_X."""
    rng = np.random.default_rng(2)
    return [rng.random(4).astype(np.float32) * 80 for _ in range(3)]


# ---------------------------------------------------------------------------
# Minimal model config
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    """Minimal training config for fast tests."""
    return {
        "units": 16,
        "dense_units": 8,
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 8,
        "epochs": 1,
        "max_sequence_length": None,
        "validation_split": 0.2,
        "patience_early_stop": 3,
        "patience_lr_reduce": 2,
        "use_early_stop": False,
    }
