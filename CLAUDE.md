# CLAUDE.md

## Project Overview

ML project for predicting Remaining Useful Life (RUL) of NASA turbofan engines using the N-CMAPSS dataset. Unified pipeline with 15 neural network architectures, automatic data loading, and W&B experiment tracking. Best model: CNN-GRU + asymmetric loss (RMSE 6.44, R² 0.91, Acc@20 99.1%). See [EXPERIMENTS.md](EXPERIMENTS.md) for full results.

## Quick Commands

```bash
# Setup
make install-dev                                    # Install all deps (uv sync)

# Code quality (run before every commit)
make check                                          # Lint + typecheck
make format                                         # Auto-format with Black

# Training
make train                                          # Train default (LSTM, 30 epochs)
python train_model.py --list-models                 # List all 15 architectures
python train_model.py --model cnn_gru --epochs 50   # Train specific model
python train_model.py --model transformer --config transformer_large.json
python train_model.py --compare --models lstm gru transformer wavenet
python train_model.py --search-config spec.json --search-workers 4

# Inference
python predict.py --model-path models/production/cnn_gru_best.keras
python predict.py --model-path models/production/cnn_gru_best.keras --unit-id 5

# Production
WANDB_MODE=offline python train_production_model.py  # Train & save best model
python example_usage.py                              # 5 real-world usage examples
```

## Environment

- **Python** 3.8+ (CI tests 3.9, 3.10, 3.11)
- **TensorFlow/Keras** for all models
- **uv** for dependency management (`uv sync`)
- **W&B** for experiment tracking (set `WANDB_MODE=offline` for local-only)
- **Black** (line-length 100) + **MyPy** (relaxed) for code quality

## Architecture

```
src/
├── data/                      # See src/data/CLAUDE.md
│   └── load_data.py           # get_datasets(fd=1..7) -> train/val/test splits
├── models/                    # See src/models/CLAUDE.md
│   ├── architectures.py       # ModelRegistry + 12 built-in architectures
│   ├── cnn_lstm_attention.py  # 2024 SOTA CNN-LSTM-Attention implementation
│   ├── mdfa.py                # Multi-Scale Dilated Fusion Attention module
│   └── __init__.py            # Lazy imports (avoids circular deps)
├── search/                    # See src/search/CLAUDE.md
│   └── hparam_search.py       # Grid/random search with parallel execution
└── utils/                     # See src/utils/CLAUDE.md
    ├── metrics.py             # RUL metrics (RMSE, MAE, PHM score, accuracy)
    ├── training_viz.py        # Training history & prediction plots
    └── visualize.py           # Data analysis & model eval visualizations

train_model.py                 # Main CLI entry point (~1000 lines)
predict.py                     # Inference API (RULPredictor class)
train_production_model.py      # Train & save CNN-GRU for deployment
scripts/                       # Ad-hoc comparison & summary tools
```

## Key Patterns

### ModelRegistry (decorator-based)

`@ModelRegistry.register("name")` on a class inheriting `BaseModel`. Build any model via `ModelRegistry.build(name, input_shape, **kwargs)`. All models compile with `asymmetric_mse` loss (penalizes late predictions 2x). See [src/models/CLAUDE.md](src/models/CLAUDE.md) for full API.

### Data Pipeline

`get_datasets(fd=1)` returns `((dev_X, dev_y), (val_X, val_y), (test_X, test_y))`. Each X is `List[np.ndarray]` with shape `(num_cycles, timesteps, num_sensors)`. `prepare_sequences()` in `train_model.py:68` flattens into `(N, timesteps, features)` arrays. `normalize_data()` at `train_model.py:100` fits `StandardScaler` on train, transforms all splits. See [src/data/CLAUDE.md](src/data/CLAUDE.md).

### Training Pipeline

`train_model()` at `train_model.py:140` orchestrates the full loop: data prep, model build, W&B init, fit with callbacks (early stopping, LR reduction), evaluate, visualize, log to W&B. `compare_models()` at `train_model.py:497` runs multiple architectures back-to-back.

### Inference Pipeline

`RULPredictor` class in `predict.py` loads a saved model + scaler + config. `predict_single(sequence)` for one reading, `evaluate_test_set(fd)` for full evaluation with metrics. See [DEPLOYMENT.md](DEPLOYMENT.md) for full API docs.

## How to Add a New Model Architecture

1. Define class in `src/models/architectures.py` (or new file, imported in `architectures.py`)
2. Inherit from `BaseModel` (line 84)
3. Decorate with `@ModelRegistry.register("your_name")`
4. Implement `static build(input_shape, units, dense_units, dropout_rate, learning_rate) -> keras.Model`
5. Call `BaseModel.compile_model(model, learning_rate)` to compile with asymmetric_mse
6. Update `get_model_info()` dict at `architectures.py:939`
7. Update `get_model_recommendations()` dict at `architectures.py:965` if applicable
8. Test: `python train_model.py --model your_name --epochs 5`
9. Run `make check`

See [src/models/CLAUDE.md](src/models/CLAUDE.md) for detailed patterns and custom layer inventory.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) on push/PR to main:
- Black formatting check
- MyPy type checking
- Matrix: Python 3.9, 3.10, 3.11
- Tool: uv for dependency installation

Always run `make check` before committing.

## Development Workflow

1. **Create a GitHub issue** with a full description of the work
2. **Create a feature branch** from main
3. **Develop and commit** — run `make check` before each commit
4. **Open a PR** referencing the issue
5. **Wait for CI to pass** before merging
6. **Merge** the PR into main

## Related Documentation

- [README.md](README.md) — Quick start guide
- [README_PRODUCTION.md](README_PRODUCTION.md) — 3-step production deployment
- [DEPLOYMENT.md](DEPLOYMENT.md) — Full deployment guide, Python API, troubleshooting
- [EXPERIMENTS.md](EXPERIMENTS.md) — SOTA architecture comparison (CNN-GRU winner)

## Common Pitfalls

- `train_model.py` is ~1000 lines — training logic lives here, NOT in `src/models/`
- `src/models/__init__.py` uses lazy imports to avoid circular deps during CLI startup
- Data downloads ~1GB on first use; cached in `data/raw/`
- Full sequences are ~20k timesteps; truncate to 1000 (`max_sequence_length`) to avoid OOM
- CNN-LSTM architecture does **not** converge; CNN-GRU is the stable hybrid variant
- W&B requires `WANDB_API_KEY` env var, or set `WANDB_MODE=offline` for local runs
- Larger models overfit on this dataset — 64 units outperforms 128 units (see [EXPERIMENTS.md](EXPERIMENTS.md))
