"""
Unit tests for RUL prediction metrics.

Tests all metric functions to ensure correct calculations and edge case handling.
"""

import numpy as np
import pytest
from src.utils.metrics import (
    rmse,
    mape,
    phm_score,
    phm_score_normalized,
    asymmetric_loss,
    rul_accuracy,
    normalized_rmse,
    normalized_mae,
    compute_all_metrics,
    format_metrics,
    compare_models,
)


class TestBasicMetrics:
    """Test basic regression metrics."""

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_calculation(self):
        """Test RMSE calculation with known values."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])
        expected_rmse = np.sqrt(np.mean([4, 4, 4, 4]))  # 2.0
        assert abs(rmse(y_true, y_pred) - expected_rmse) < 1e-6

    def test_mape_perfect_prediction(self):
        """Test MAPE with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        assert mape(y_true, y_pred) == 0.0

    def test_mape_calculation(self):
        """Test MAPE calculation with known values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # (|100-110|/100 + |200-190|/200) / 2 * 100 = (0.1 + 0.05) / 2 * 100 = 7.5
        expected_mape = 7.5
        assert abs(mape(y_true, y_pred) - expected_mape) < 1e-6

    def test_mape_handles_zero(self):
        """Test MAPE handles zero true values with epsilon."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 10.0])
        # Should not raise division by zero error
        result = mape(y_true, y_pred)
        assert np.isfinite(result)


class TestPHMScore:
    """Test PHM Society scoring function."""

    def test_phm_score_perfect_prediction(self):
        """Test PHM score with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert phm_score(y_true, y_pred) == 0.0

    def test_phm_score_early_predictions(self):
        """Test PHM score penalizes early predictions less."""
        y_true = np.array([20.0])
        y_pred_early = np.array([10.0])  # 10 cycles early (d = -10)
        score_early = phm_score(y_true, y_pred_early)
        # exp(-(-10)/13) - 1 = exp(0.769) - 1 ≈ 1.16
        assert 1.0 < score_early < 1.5

    def test_phm_score_late_predictions(self):
        """Test PHM score penalizes late predictions more heavily."""
        y_true = np.array([20.0])
        y_pred_late = np.array([30.0])  # 10 cycles late (d = 10)
        score_late = phm_score(y_true, y_pred_late)
        # exp(10/10) - 1 = exp(1) - 1 ≈ 1.72
        assert 1.5 < score_late < 2.0

    def test_phm_score_asymmetric_penalty(self):
        """Test that late predictions are penalized more than early ones."""
        y_true = np.array([50.0])
        y_pred_early = np.array([40.0])  # 10 cycles early
        y_pred_late = np.array([60.0])   # 10 cycles late

        score_early = phm_score(y_true, y_pred_early)
        score_late = phm_score(y_true, y_pred_late)

        assert score_late > score_early

    def test_phm_score_normalized(self):
        """Test normalized PHM score returns per-sample average."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        total_score = phm_score(y_true, y_pred)
        normalized = phm_score_normalized(y_true, y_pred)
        assert abs(normalized - (total_score / 3)) < 1e-6


class TestAsymmetricLoss:
    """Test asymmetric loss function."""

    def test_asymmetric_loss_perfect_prediction(self):
        """Test asymmetric loss with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert asymmetric_loss(y_true, y_pred) == 0.0

    def test_asymmetric_loss_late_penalty(self):
        """Test asymmetric loss penalizes late predictions more."""
        y_true = np.array([50.0])
        y_pred_early = np.array([40.0])  # 10 cycles early (error = -10)
        y_pred_late = np.array([60.0])   # 10 cycles late (error = 10)

        loss_early = asymmetric_loss(y_true, y_pred_early, alpha=2.0)
        loss_late = asymmetric_loss(y_true, y_pred_late, alpha=2.0)

        # Late: alpha * 10^2 = 2 * 100 = 200
        # Early: 10^2 = 100
        assert loss_late == 2 * loss_early

    def test_asymmetric_loss_alpha_parameter(self):
        """Test asymmetric loss with different alpha values."""
        y_true = np.array([50.0])
        y_pred = np.array([60.0])  # 10 cycles late

        loss_alpha_1 = asymmetric_loss(y_true, y_pred, alpha=1.0)
        loss_alpha_2 = asymmetric_loss(y_true, y_pred, alpha=2.0)
        loss_alpha_3 = asymmetric_loss(y_true, y_pred, alpha=3.0)

        assert loss_alpha_2 > loss_alpha_1
        assert loss_alpha_3 > loss_alpha_2


class TestRULAccuracy:
    """Test RUL accuracy metric."""

    def test_rul_accuracy_perfect(self):
        """Test RUL accuracy with all predictions within threshold."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])  # All within ±10
        assert rul_accuracy(y_true, y_pred, threshold=10.0) == 100.0

    def test_rul_accuracy_none(self):
        """Test RUL accuracy with no predictions within threshold."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([25.0, 35.0, 45.0, 55.0])  # All off by ±15
        assert rul_accuracy(y_true, y_pred, threshold=10.0) == 0.0

    def test_rul_accuracy_partial(self):
        """Test RUL accuracy with some predictions within threshold."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 45.0, 55.0])  # 2 out of 4 within ±10
        assert rul_accuracy(y_true, y_pred, threshold=10.0) == 50.0

    def test_rul_accuracy_custom_threshold(self):
        """Test RUL accuracy with different thresholds."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])

        acc_5 = rul_accuracy(y_true, y_pred, threshold=5.0)
        acc_10 = rul_accuracy(y_true, y_pred, threshold=10.0)

        assert acc_10 >= acc_5  # Larger threshold should have >= accuracy


class TestNormalizedMetrics:
    """Test normalized RMSE and MAE metrics."""

    def test_normalized_rmse_perfect(self):
        """Test normalized RMSE with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        result = normalized_rmse(y_true, y_pred, y_min=0.0, y_max=65.0)
        assert result == 0.0

    def test_normalized_mae_perfect(self):
        """Test normalized MAE with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        result = normalized_mae(y_true, y_pred, y_min=0.0, y_max=65.0)
        assert result == 0.0

    def test_normalized_rmse_calculation(self):
        """Test normalized RMSE calculation with known values."""
        y_true = np.array([0.0, 65.0])
        y_pred = np.array([0.0, 65.0])
        y_min, y_max = 0.0, 65.0

        # Perfect prediction → normalized RMSE = 0
        result = normalized_rmse(y_true, y_pred, y_min, y_max)
        assert result == 0.0

        # Test with error
        y_pred_err = np.array([6.5, 58.5])  # Errors: 6.5, 6.5
        # Normalized: [0, 1], [0.1, 0.9]
        # RMSE = sqrt(mean([0.1^2, 0.1^2])) = 0.1
        result_err = normalized_rmse(y_true, y_pred_err, y_min, y_max)
        assert abs(result_err - 0.1) < 1e-6

    def test_normalized_mae_calculation(self):
        """Test normalized MAE calculation with known values."""
        y_true = np.array([0.0, 65.0])
        y_pred = np.array([6.5, 58.5])  # Errors: 6.5, 6.5
        y_min, y_max = 0.0, 65.0

        # Normalized errors: [0.1, 0.1]
        # MAE = mean([0.1, 0.1]) = 0.1
        result = normalized_mae(y_true, y_pred, y_min, y_max)
        assert abs(result - 0.1) < 1e-6

    def test_normalized_metrics_zero_range(self):
        """Test normalized metrics handle zero range gracefully."""
        y_true = np.array([50.0, 50.0])
        y_pred = np.array([50.0, 50.0])

        # When y_min == y_max, should return 0
        result_rmse = normalized_rmse(y_true, y_pred, y_min=50.0, y_max=50.0)
        result_mae = normalized_mae(y_true, y_pred, y_min=50.0, y_max=50.0)

        assert result_rmse == 0.0
        assert result_mae == 0.0

    def test_normalized_metrics_scale_invariance(self):
        """Test that normalized metrics are scale-invariant."""
        # Same relative errors should give same normalized metric
        y_true_1 = np.array([0.0, 100.0])
        y_pred_1 = np.array([10.0, 90.0])  # 10% errors

        y_true_2 = np.array([0.0, 10.0])
        y_pred_2 = np.array([1.0, 9.0])    # 10% errors

        rmse_1 = normalized_rmse(y_true_1, y_pred_1, y_min=0.0, y_max=100.0)
        rmse_2 = normalized_rmse(y_true_2, y_pred_2, y_min=0.0, y_max=10.0)

        assert abs(rmse_1 - rmse_2) < 1e-6


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""

    def test_compute_all_metrics_basic(self):
        """Test compute_all_metrics returns all expected metrics."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])

        metrics = compute_all_metrics(y_true, y_pred)

        required_keys = [
            "mse", "rmse", "mae", "mape", "r2",
            "phm_score", "phm_score_normalized",
            "asymmetric_loss",
            "accuracy_10", "accuracy_15", "accuracy_20",
        ]

        for key in required_keys:
            assert key in metrics
            assert np.isfinite(metrics[key])

    def test_compute_all_metrics_with_normalization(self):
        """Test compute_all_metrics includes normalized metrics when provided."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])

        metrics = compute_all_metrics(y_true, y_pred, y_min=0.0, y_max=65.0)

        assert "rmse_normalized" in metrics
        assert "mae_normalized" in metrics
        assert np.isfinite(metrics["rmse_normalized"])
        assert np.isfinite(metrics["mae_normalized"])

    def test_compute_all_metrics_without_normalization(self):
        """Test compute_all_metrics without normalization params."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert "rmse_normalized" not in metrics
        assert "mae_normalized" not in metrics

    def test_compute_all_metrics_perfect_prediction(self):
        """Test all metrics with perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])

        metrics = compute_all_metrics(y_true, y_pred, y_min=0.0, y_max=65.0)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mape"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["phm_score"] == 0.0
        assert metrics["asymmetric_loss"] == 0.0
        assert metrics["rmse_normalized"] == 0.0
        assert metrics["mae_normalized"] == 0.0
        assert metrics["accuracy_10"] == 100.0
        assert metrics["accuracy_20"] == 100.0


class TestFormatMetrics:
    """Test format_metrics function."""

    def test_format_metrics_output(self):
        """Test format_metrics returns a string."""
        metrics = {
            "mse": 100.0,
            "rmse": 10.0,
            "mae": 8.0,
            "mape": 5.0,
            "r2": 0.95,
            "phm_score": 50.0,
            "phm_score_normalized": 2.5,
            "asymmetric_loss": 120.0,
            "accuracy_10": 85.0,
            "accuracy_15": 92.0,
            "accuracy_20": 98.0,
        }

        result = format_metrics(metrics)

        assert isinstance(result, str)
        assert "RMSE" in result
        assert "MAE" in result
        assert "R2 Score" in result
        assert "10.0" in result  # RMSE value

    def test_format_metrics_with_normalized(self):
        """Test format_metrics includes normalized metrics when present."""
        metrics = {
            "mse": 100.0,
            "rmse": 10.0,
            "mae": 8.0,
            "mape": 5.0,
            "r2": 0.95,
            "phm_score": 50.0,
            "phm_score_normalized": 2.5,
            "asymmetric_loss": 120.0,
            "accuracy_10": 85.0,
            "accuracy_15": 92.0,
            "accuracy_20": 98.0,
            "rmse_normalized": 0.045,
            "mae_normalized": 0.038,
        }

        result = format_metrics(metrics)

        assert "normalized" in result.lower()
        assert "0.045" in result


class TestCompareModels:
    """Test compare_models function."""

    def test_compare_models_rmse(self):
        """Test compare_models finds best model by RMSE."""
        results = {
            "model_a": {"rmse": 10.0, "mae": 8.0, "r2": 0.90},
            "model_b": {"rmse": 8.0, "mae": 7.0, "r2": 0.92},
            "model_c": {"rmse": 12.0, "mae": 9.0, "r2": 0.88},
        }

        best_model, best_metrics = compare_models(results, primary_metric="rmse")

        assert best_model == "model_b"
        assert best_metrics["rmse"] == 8.0

    def test_compare_models_r2(self):
        """Test compare_models finds best model by R2 (higher is better)."""
        results = {
            "model_a": {"rmse": 10.0, "mae": 8.0, "r2": 0.90},
            "model_b": {"rmse": 8.0, "mae": 7.0, "r2": 0.92},
            "model_c": {"rmse": 12.0, "mae": 9.0, "r2": 0.88},
        }

        best_model, best_metrics = compare_models(results, primary_metric="r2")

        assert best_model == "model_b"
        assert best_metrics["r2"] == 0.92

    def test_compare_models_accuracy(self):
        """Test compare_models handles accuracy metrics (higher is better)."""
        results = {
            "model_a": {"rmse": 10.0, "accuracy_10": 85.0},
            "model_b": {"rmse": 8.0, "accuracy_10": 90.0},
            "model_c": {"rmse": 12.0, "accuracy_10": 80.0},
        }

        best_model, best_metrics = compare_models(results, primary_metric="accuracy_10")

        assert best_model == "model_b"
        assert best_metrics["accuracy_10"] == 90.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test metrics handle empty arrays gracefully."""
        y_true = np.array([])
        y_pred = np.array([])

        # Most metrics should handle empty arrays
        # (may return NaN or raise error depending on implementation)
        with pytest.raises((ValueError, RuntimeWarning)):
            rmse(y_true, y_pred)

    def test_single_sample(self):
        """Test metrics work with single sample."""
        y_true = np.array([50.0])
        y_pred = np.array([48.0])

        result = rmse(y_true, y_pred)
        assert result == 2.0

    def test_negative_values(self):
        """Test metrics handle negative RUL values (shouldn't happen in practice)."""
        y_true = np.array([-10.0, 20.0])
        y_pred = np.array([-8.0, 22.0])

        # Metrics should still calculate correctly
        result = rmse(y_true, y_pred)
        assert np.isfinite(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
