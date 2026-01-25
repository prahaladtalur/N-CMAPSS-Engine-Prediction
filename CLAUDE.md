# CLAUDE.md

## Project Overview

This is an ML project for predicting Remaining Useful Life (RUL) of NASA turbofan engines using the N-CMAPSS dataset. It provides a unified pipeline with 14 neural network architectures, automatic data loading, and W&B experiment tracking.

## Development Workflow

1. **Create a GitHub issue** with a full description of the work to be done
2. **Create a feature branch** from main to work on the issue
3. **Develop and commit** changes to the branch
4. **Open a PR** when work is complete, referencing the issue
5. **Wait for CI to pass** before merging
6. **Merge** the PR into main

## Quick Commands

```bash
# Install dependencies
make install-dev   # or: uv sync

# Run all checks before committing
make check         # runs lint + typecheck

# Format code
make format

# Train a model
make train         # or: python train_model.py --model lstm --epochs 30

# List all available models
python train_model.py --list-models

# Compare multiple models
python train_model.py --compare --models lstm gru transformer
```

## CI/CD

GitHub Actions runs on every push/PR to main:
- **Lint**: Black formatting check
- **Typecheck**: MyPy static analysis
- **Matrix**: Python 3.9, 3.10, 3.11

Run `make check` locally before pushing to catch issues early.

## Architecture

```
src/
├── data/load_data.py      # Single data loader using rul_datasets library
├── models/
│   ├── architectures.py   # 14 registered models (LSTM, GRU, TCN, Transformer, etc.)
│   └── __init__.py        # Exports ModelRegistry and train functions
├── search/hparam_search.py # Grid/random hyperparameter search
└── utils/
    ├── metrics.py         # RUL metrics (PHM scoring, RMSE, MAE)
    └── training_viz.py    # Training plots and visualizations

train_model.py             # Main CLI entry point
scripts/                   # Helper scripts for comparisons
```

## Key Patterns

- **ModelRegistry**: Models registered via `@ModelRegistry.register("name")` decorator
- **Single entry point**: All training goes through `train_model.py`
- **W&B integration**: Automatic logging of metrics, hyperparameters, visualizations
- **Data auto-download**: First run automatically downloads and caches the dataset

## Available Models

LSTM, BiLSTM, GRU, BiGRU, attention_lstm, resnet_lstm, TCN, WaveNet, cnn_lstm, cnn_gru, inception_lstm, transformer, MLP

## Code Style

- Black formatter (line-length: 100)
- Python 3.8+ required
- MyPy for type checking (relaxed settings)
