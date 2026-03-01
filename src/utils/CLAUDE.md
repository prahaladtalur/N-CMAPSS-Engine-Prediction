# src/utils/ — Metrics & Visualization

## Purpose

Evaluation metrics for RUL prediction and visualization utilities for training, data analysis, and model evaluation. All public functions are re-exported via `__init__.py` — import directly from `src.utils`.

## Key Files

### metrics.py — Evaluation Metrics

**Core function**: `compute_all_metrics(y_true, y_pred, y_min=None, y_max=None) -> dict`
Returns all metrics in one call. Pass `y_min`/`y_max` for normalized metrics (used for paper comparison).

| Function | Purpose | Note |
|----------|---------|------|
| `rmse(y_true, y_pred)` | Root Mean Squared Error | Primary ranking metric |
| `mape(y_true, y_pred)` | Mean Absolute Percentage Error | |
| `phm_score(y_true, y_pred)` | Official PHM08 Challenge scoring | Lower is better; asymmetric |
| `phm_score_normalized(y_true, y_pred)` | Per-sample average PHM | Easier to interpret |
| `asymmetric_loss(y_true, y_pred, alpha)` | NumPy version of training loss | For evaluation only |
| `rul_accuracy(y_true, y_pred, threshold)` | % predictions within threshold | Default threshold=10 |
| `normalized_rmse(y_true, y_pred, y_min, y_max)` | RMSE on [0,1]-scaled RUL | For SOTA comparison |
| `normalized_mae(y_true, y_pred, y_min, y_max)` | MAE on [0,1]-scaled RUL | For SOTA comparison |
| `compare_models(results_dict, primary_metric)` | Find best model from results | |
| `format_metrics(metrics_dict)` | Pretty-print metrics string | |

### training_viz.py — Training Visualization

| Function | Purpose |
|----------|---------|
| `plot_training_history(history, model_name, save_path)` | Loss/accuracy curves |
| `plot_predictions(y_true, y_pred, model_name, save_path)` | Actual vs predicted scatter |
| `plot_error_distribution(y_true, y_pred, model_name, save_path)` | Error histogram |
| `plot_model_comparison(results, metrics, save_path)` | Multi-model side-by-side bars |
| `plot_sample_predictions(y_true, y_pred, model_name, save_path)` | Sample-level predictions |
| `create_evaluation_report(model_name, metrics, y_true, y_pred, history, save_dir)` | Full report to directory |

### visualize.py — Data Analysis & Model Eval (28KB)

**Data analysis**: `plot_sensor_degradation`, `plot_sensor_correlation_heatmap`, `plot_multi_sensor_lifecycle`

**Model evaluation**: `plot_rul_trajectory`, `plot_critical_zone_analysis`, `plot_prediction_confidence`

**Basic**: `plot_rul_distribution`, `plot_sensor_time_series`, `visualize_dataset`

## Import Pattern

All functions re-exported via `__init__.py`:
```python
from src.utils import compute_all_metrics, plot_training_history, rmse
```

## Adding a New Metric

1. Add function in `metrics.py` following existing signature `(y_true, y_pred, ...) -> float`
2. Add to `compute_all_metrics()` return dict if it should always be computed
3. Add to `format_metrics()` for display formatting
4. Export in `__init__.py`
5. Run `make check`

## Adding a New Visualization

1. Add function in `training_viz.py` (training-related) or `visualize.py` (data/eval)
2. Follow pattern: accept `save_path=None`, call `plt.savefig()` if provided, else `plt.show()`
3. Export in `__init__.py`
4. Run `make check`

## PHM Score Explained

The PHM08 scoring function is the standard metric for RUL prediction competitions. It is exponentially asymmetric: late predictions (thinking the engine has more life than it does) are penalized ~3x more than early predictions at the same error magnitude. This aligns with the safety-critical nature of the domain.

```
s_i = exp(-d/13) - 1   if d < 0  (early prediction — less severe)
s_i = exp(d/10)  - 1   if d >= 0 (late prediction — dangerous)
```

See [../models/CLAUDE.md](../models/CLAUDE.md) for the corresponding `asymmetric_mse` training loss.
