# N-CMAPSS Engine RUL Prediction

Predict remaining useful life (RUL) for NASA's turbofan engines with a single, opinionated pipeline. The repository now revolves around **one data loader, one training entry point, and one set of utilities**, so you can find what you need without bouncing between duplicate modules or docs.

## Workflow at a Glance

```bash
# 1) Install everything (creates a virtual env if needed)
pip install uv
uv sync

# 2) Train a model (dataset downloads/caches automatically on first run)
python train_model.py --model lstm --epochs 30

# 3) Compare architectures when you're ready
python train_model.py --compare --models lstm gru transformer
```

The CLI handles:
- downloading & caching any FD00X split through `src/data/load_data.py`
- preparing sequences, normalization, wandb logging, and result plots via `train_model.py`
- switching among all registered architectures defined in `src/models/architectures.py`

## Feature Highlights

- **Single source of truth for data** – `src/data/load_data.py` sets up RUL datasets and returns `(dev, val, test)` splits ready for modeling.
- **Curated model zoo** – `src/models/architectures.py` registers LSTM/GRU variants, CNN hybrids, TCN/WaveNet blocks, and a Transformer encoder; use `train_model.py --list-models` to inspect names.
- **Consistent training loop** – `src/models/train.py` exposes `train_model(...)`, `compare_models(...)`, and `prepare_sequences(...)`, removing the legacy LSTM-only helpers.
- **Evaluation utilities** – Metrics (`src/utils/metrics.py`) and visualization helpers (`src/utils/training_viz.py`, `src/utils/visualize.py`) cover both dataset analysis and post-training reporting.

## Project Layout

```
N-CMAPSS-Engine-Prediction
├── src/
│   ├── data/                # dataset download & caching
│   ├── models/              # model registry + training pipeline
│   └── utils/               # metrics and visualization helpers
├── train_model.py           # main CLI (use this)
├── scripts/                 # comparison + reporting helpers
├── pyproject.toml / uv.lock
```

## Key Commands

| Command | Why you run it |
| --- | --- |
| `python train_model.py --list-models` | View every registered architecture with short descriptions |
| `python train_model.py --model <name> [--epochs 50 --units 128 ...]` | Train one model with custom hyperparameters |
| `python train_model.py --compare --models lstm gru tcn` | Fit multiple architectures back-to-back and write comparison plots under `results/comparison/` |
| `python scripts/compare_saved_runs.py` | Build a comparison plot from saved runs (no retraining) |
| `python scripts/make_best_model_summary.py` | Create a single summary image for the best model |

All commands accept `--fd 1..7` to switch datasets. Feature normalization and visual outputs can be toggled with `--no-normalize` / `--no-visualize` during training.

If you only remember one thing: run `python train_model.py --model lstm` to bootstrap everything. From there you can branch into comparisons, more advanced architectures, or the visualization suite without relearning new entry points.

## Evaluation Metrics

The project tracks comprehensive metrics for RUL prediction, including both standard regression metrics and domain-specific RUL metrics. All metrics are automatically logged to Weights & Biases for experiment tracking.

### Standard Regression Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MSE** | Mean Squared Error | Average squared difference between predictions and true values. Lower is better. |
| **RMSE** | Root Mean Squared Error | Square root of MSE, in same units as RUL (cycles). Lower is better. Typical range: 5-15 cycles for good models. |
| **MAE** | Mean Absolute Error | Average absolute difference between predictions and true values. Lower is better. More interpretable than RMSE. |
| **MAPE** | Mean Absolute Percentage Error | Percentage error relative to true values. Lower is better. Useful for understanding relative accuracy. |
| **R² Score** | Coefficient of Determination | Proportion of variance explained by the model. Range: [0, 1], higher is better. >0.90 is excellent. |

### RUL-Specific Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **PHM Score** | PHM Society Challenge scoring function | Asymmetric penalty: late predictions (predicting failure after actual) penalized more heavily than early predictions. Lower is better, 0 = perfect. |
| **Asymmetric Loss** | Custom loss with α penalty for late predictions | Penalizes late predictions by factor α (default 2.0). Used during training to improve safety. Lower is better. |
| **Accuracy@N** | Percentage of predictions within ±N cycles | Practical metric: % of predictions within acceptable threshold. Typical thresholds: 10, 15, 20 cycles. Higher is better. |

### Normalized Metrics (Paper Comparison)

To enable fair comparison with published research, we also compute **normalized metrics** by scaling RUL values to [0, 1]:

| Metric | Description | SOTA Benchmark (MDFA Paper) |
|--------|-------------|------------------------------|
| **RMSE (normalized)** | RMSE on RUL scaled to [0, 1] | **0.021–0.032** (target) |
| **MAE (normalized)** | MAE on RUL scaled to [0, 1] | **0.018–0.026** (target) |
| **R² (normalized)** | R² on normalized values | **0.987–0.995** (target) |

**Why normalize?** Different papers use different RUL ranges (max RUL varies by dataset). Normalizing to [0, 1] allows direct comparison across studies. A normalized RMSE of 0.03 means predictions are off by ~3% of the full RUL range.

**Current Best:** Our CNN-GRU model achieves RMSE_norm = 0.098, which is ~3× the SOTA target, indicating room for architectural improvements (see Issue #8).

### Understanding the Metrics

**When is a model "good"?**
- **RMSE < 10 cycles**: Good for practical use
- **MAE < 8 cycles**: Consistently accurate predictions
- **R² > 0.90**: Model explains >90% of variance
- **Accuracy@20 > 95%**: Most predictions within acceptable range
- **RMSE_norm < 0.05**: Approaching research SOTA

**Why PHM Score and Asymmetric Loss?**
In predictive maintenance, **late predictions are more dangerous** than early ones:
- **Early prediction** (predict failure at cycle 90, actual failure at cycle 100): Unnecessary maintenance, but safe
- **Late prediction** (predict failure at cycle 110, actual failure at cycle 100): Catastrophic failure, unsafe

PHM Score and Asymmetric Loss mathematically encode this safety preference by penalizing late predictions more heavily.

### Metrics in Weights & Biases

All metrics are automatically logged to W&B with the following structure:

```
test/rmse                    # Raw RMSE (cycles)
test/mae                     # Raw MAE (cycles)
test/r2                      # R² score
test/rmse_normalized         # Normalized RMSE [0, 1]
test/mae_normalized          # Normalized MAE [0, 1]
test/rmse_normalized_gap     # Gap from SOTA (multiplier)
test/mae_normalized_gap      # Gap from SOTA (multiplier)
test/phm_score_normalized    # Per-sample PHM score
test/accuracy_10             # Accuracy within ±10 cycles
test/accuracy_20             # Accuracy within ±20 cycles

sota/rmse_normalized_target  # SOTA benchmark (0.032)
sota/mae_normalized_target   # SOTA benchmark (0.026)

charts/training_history      # Loss & MAE over epochs
charts/predictions           # True vs predicted scatter plot
charts/error_distribution    # Histogram of prediction errors

predictions_table            # Interactive table with 500 sample predictions
```

**Run Summary:** Each W&B run includes comprehensive metadata (model architecture, parameters, dataset info, training config) for easy filtering and comparison.

### Running Tests

Metric calculations are validated with comprehensive unit tests:

```bash
# Run all tests
make test

# Or directly with pytest
pytest tests/test_metrics.py -v
```

Tests cover:
- Perfect predictions (all metrics should be 0 or optimal)
- Known values (verify calculation correctness)
- Edge cases (empty arrays, single samples, zero values)
- PHM score asymmetry (late > early penalty)
- Normalized metric scale invariance
