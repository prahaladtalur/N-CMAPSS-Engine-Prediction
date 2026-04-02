"""
Feature preprocessing helpers shared by training and inference.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from src.data.load_data import W_CHANNELS


class OperatingConditionResidualizer:
    """Remove sensor variation explained by operating conditions and flight phase."""

    def __init__(
        self,
        operating_condition_indices: Sequence[int] = W_CHANNELS,
        *,
        include_time_position: bool = True,
        fit_healthy_only: bool = True,
    ):
        self.operating_condition_indices = tuple(int(idx) for idx in operating_condition_indices)
        self.include_time_position = bool(include_time_position)
        self.fit_healthy_only = bool(fit_healthy_only)
        self.target_indices_: list[int] = []
        self.coefficients_: Optional[np.ndarray] = None
        self.healthy_rul_threshold_: Optional[float] = None

    def _build_design(self, values: np.ndarray) -> np.ndarray:
        operating = values[..., list(self.operating_condition_indices)]
        operating_flat = operating.reshape(-1, len(self.operating_condition_indices)).astype(
            np.float32
        )
        design_terms = [operating_flat]

        if self.include_time_position:
            if values.ndim == 2:
                time_position = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)[
                    :, None
                ]
            else:
                time_axis = np.linspace(0.0, 1.0, num=values.shape[1], dtype=np.float32)
                time_position = np.broadcast_to(time_axis, values.shape[:-1]).reshape(-1, 1)
            design_terms.append(time_position)

        design_terms.append(np.ones((operating_flat.shape[0], 1), dtype=np.float32))
        return np.concatenate(design_terms, axis=1)

    def fit(
        self,
        units: Sequence[np.ndarray],
        labels: Optional[Sequence[np.ndarray]] = None,
        healthy_rul_threshold: Optional[float] = None,
    ) -> "OperatingConditionResidualizer":
        """Fit a dev-split baseline using operating conditions from mostly healthy cycles."""
        if not units:
            raise ValueError("Cannot fit OperatingConditionResidualizer on an empty split")
        if labels is not None and len(labels) != len(units):
            raise ValueError("labels must have the same number of units as features")

        num_features = int(units[0].shape[-1])
        operating_index_set = set(self.operating_condition_indices)
        self.target_indices_ = [
            idx for idx in range(num_features) if idx not in operating_index_set
        ]
        if not self.target_indices_:
            design_width = (
                len(self.operating_condition_indices) + 1 + int(self.include_time_position)
            )
            self.coefficients_ = np.zeros((design_width, 0), dtype=np.float32)
            return self

        fit_units = list(units)
        if labels is not None and self.fit_healthy_only:
            threshold = healthy_rul_threshold
            if threshold is None:
                threshold = float(max(np.max(unit_labels) for unit_labels in labels))
            self.healthy_rul_threshold_ = float(threshold)

            healthy_units = []
            for unit, unit_labels in zip(units, labels):
                healthy_mask = np.asarray(unit_labels) >= threshold
                if np.any(healthy_mask):
                    healthy_units.append(np.asarray(unit)[healthy_mask])
            if healthy_units:
                fit_units = healthy_units

        operating_stack = np.concatenate(
            [self._build_design(np.asarray(unit, dtype=np.float32)) for unit in fit_units],
            axis=0,
        )
        target_stack = np.concatenate(
            [
                np.asarray(unit, dtype=np.float32)[..., self.target_indices_].reshape(
                    -1, len(self.target_indices_)
                )
                for unit in fit_units
            ],
            axis=0,
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            operating_stack, target_stack.astype(np.float32), rcond=None
        )
        self.coefficients_ = coefficients.astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Replace non-W channels with residuals against the fitted baseline."""
        if self.coefficients_ is None:
            raise ValueError("OperatingConditionResidualizer must be fitted before transform()")

        data = np.asarray(values, dtype=np.float32)
        if data.ndim not in (2, 3):
            raise ValueError("Expected a 2D sequence or 3D batch of sequences")

        transformed = data.copy()
        if not self.target_indices_:
            return transformed

        design = self._build_design(transformed)
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
