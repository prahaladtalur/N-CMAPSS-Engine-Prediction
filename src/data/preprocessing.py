"""
Feature preprocessing helpers shared by training and inference.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from src.data.load_data import W_CHANNELS


class OperatingConditionResidualizer:
    """Remove sensor variation explained by the operating-condition channels."""

    def __init__(self, operating_condition_indices: Sequence[int] = W_CHANNELS):
        self.operating_condition_indices = tuple(int(idx) for idx in operating_condition_indices)
        self.target_indices_: list[int] = []
        self.coefficients_: Optional[np.ndarray] = None

    def fit(self, units: Sequence[np.ndarray]) -> "OperatingConditionResidualizer":
        """Fit a linear sensor baseline using dev-split operating conditions only."""
        if not units:
            raise ValueError("Cannot fit OperatingConditionResidualizer on an empty split")

        num_features = int(units[0].shape[-1])
        operating_index_set = set(self.operating_condition_indices)
        self.target_indices_ = [idx for idx in range(num_features) if idx not in operating_index_set]
        if not self.target_indices_:
            self.coefficients_ = np.zeros((len(self.operating_condition_indices) + 1, 0), dtype=np.float32)
            return self

        operating_stack = np.concatenate(
            [
                unit[..., list(self.operating_condition_indices)].reshape(
                    -1, len(self.operating_condition_indices)
                )
                for unit in units
            ],
            axis=0,
        )
        target_stack = np.concatenate(
            [unit[..., self.target_indices_].reshape(-1, len(self.target_indices_)) for unit in units],
            axis=0,
        )
        design = np.concatenate(
            [operating_stack.astype(np.float32), np.ones((operating_stack.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, target_stack.astype(np.float32), rcond=None)
        self.coefficients_ = coefficients.astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Replace non-W channels with residuals against the fitted operating baseline."""
        if self.coefficients_ is None:
            raise ValueError("OperatingConditionResidualizer must be fitted before transform()")

        data = np.asarray(values, dtype=np.float32)
        if data.ndim not in (2, 3):
            raise ValueError("Expected a 2D sequence or 3D batch of sequences")

        transformed = data.copy()
        if not self.target_indices_:
            return transformed

        operating = transformed[..., list(self.operating_condition_indices)]
        operating_flat = operating.reshape(-1, len(self.operating_condition_indices))
        design = np.concatenate(
            [operating_flat, np.ones((operating_flat.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        baseline = design @ self.coefficients_
        transformed[..., self.target_indices_] -= baseline.reshape(
            transformed.shape[:-1] + (len(self.target_indices_),)
        )
        return transformed


def transform_feature_array(
    values: np.ndarray,
    *,
    residualizer: Optional[OperatingConditionResidualizer] = None,
    scaler: Optional[Any] = None,
) -> np.ndarray:
    """Apply residualization and feature scaling to 2D/3D feature arrays."""
    transformed = np.asarray(values, dtype=np.float32)

    if residualizer is not None:
        transformed = residualizer.transform(transformed)

    if scaler is None:
        return transformed.astype(np.float32, copy=False)

    if transformed.ndim == 2:
        return scaler.transform(transformed).astype(np.float32)

    if transformed.ndim == 3:
        original_shape = transformed.shape
        flattened = transformed.reshape(-1, original_shape[-1])
        flattened = scaler.transform(flattened)
        return flattened.reshape(original_shape).astype(np.float32)

    raise ValueError("Expected a 2D sequence or 3D batch of sequences")
