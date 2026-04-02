"""
Prediction post-processing helpers shared by training and inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def causal_exponential_moving_average(
    values: np.ndarray,
    *,
    decay: float = 0.9,
    window: Optional[int] = 100,
) -> np.ndarray:
    """Smooth a 1D prediction trace using a causal exponentially weighted average."""
    if decay <= 0.0 or decay > 1.0:
        raise ValueError("decay must be in the interval (0, 1]")

    series = np.asarray(values, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError("Expected a 1D prediction trace")
    if series.size == 0:
        return series.copy()

    if window is not None and window <= 0:
        raise ValueError("window must be positive when provided")

    smoothed = np.empty_like(series)
    for idx in range(series.size):
        start_idx = 0 if window is None else max(0, idx - window + 1)
        history = series[start_idx : idx + 1]
        powers = np.arange(history.size - 1, -1, -1, dtype=np.float32)
        weights = np.power(np.float32(decay), powers)
        smoothed[idx] = np.sum(history * weights) / np.sum(weights)

    return smoothed
