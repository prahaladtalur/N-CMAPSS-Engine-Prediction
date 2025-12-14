# N-CMAPSS Engine RUL Prediction

Predict remaining useful life (RUL) for NASA's turbofan engines with a single, opinionated pipeline. The repository now revolves around **one data loader, one training entry point, and one set of utilities**, so you can find what you need without bouncing between duplicate modules or docs.

## Workflow at a Glance

```bash
# 1) Install everything (creates a virtual env if needed)
pip install uv
uv sync

# 2) Train a model (dataset downloads/caches automatically on first run)
python train_model.py --model lstm --epochs 30

# 3) Compare architectures or explore visuals when you're ready
python train_model.py --compare --models lstm gru transformer
python scripts/example_visualizations.py --data
```

The CLI handles:
- downloading & caching any FD00X split through `src/data/load_data.py`
- preparing sequences, normalization, wandb logging, and result plots via `src/models/train.py`
- switching among all registered architectures defined in `src/models/architectures.py`

## Feature Highlights

- **Single source of truth for data** – `src/data/load_data.py` sets up RUL datasets and returns `(dev, val, test)` splits ready for modeling.
- **Curated model zoo** – `src/models/architectures.py` registers LSTM/GRU variants, CNN hybrids, TCN/WaveNet blocks, and a Transformer encoder; use `train_model.py --list-models` to inspect names.
- **Consistent training loop** – `src/models/train.py` exposes `train_model(...)`, `compare_models(...)`, and `prepare_sequences(...)`, removing the legacy LSTM-only helpers.
- **Evaluation utilities** – Metrics (`src/utils/metrics.py`) and visualization helpers (`src/utils/training_viz.py`, `src/utils/visualize.py`) cover both dataset analysis and post-training reporting. The sample script in `scripts/example_visualizations.py` shows end-to-end usage.

## Project Layout

```
N-CMAPSS-Engine-Prediction
├── src/
│   ├── data/                # dataset download & caching
│   ├── models/              # model registry + training pipeline
│   └── utils/               # metrics and visualization helpers
├── train_model.py           # main CLI (use this)
├── scripts/                 # ready-made visualization walkthroughs
├── notebooks/               # exploratory work
├── requirements.txt / pyproject.toml / uv.lock
└── docs: GETTING_STARTED.md, MODEL_SELECTION.md, VISUALIZATIONS.md, CLEANUP_REPORT.md
```

## Key Commands

| Command | Why you run it |
| --- | --- |
| `python train_model.py --list-models` | View every registered architecture with short descriptions |
| `python train_model.py --model <name> [--epochs 50 --units 128 ...]` | Train one model with custom hyperparameters |
| `python train_model.py --compare --models lstm gru tcn` | Fit multiple architectures back-to-back and write comparison plots under `results/comparison/` |
| `python scripts/example_visualizations.py --basic --fd 1` | Inspect dataset distributions and sensor traces before training |
| `python scripts/example_visualizations.py --model` | Showcase post-training analysis (RUL trajectories, error histograms, etc.) |

All commands accept `--fd 1..7` to switch datasets. Feature normalization and visual outputs can be toggled with `--no-normalize` / `--no-visualize` during training.

## Documentation Map

- **Need the full walkthrough?** Read [GETTING_STARTED.md](GETTING_STARTED.md).
- **Choosing a model?** Open [MODEL_SELECTION.md](MODEL_SELECTION.md) for pros/cons and recommendations (also available via `--recommend`).
- **Curious about plots?** [VISUALIZATIONS.md](VISUALIZATIONS.md) enumerates every function showcased by `scripts/example_visualizations.py`.
- **Code cleanup details?** See [CLEANUP_REPORT.md](CLEANUP_REPORT.md) for refactoring notes and improvements.

---

If you only remember one thing: run `python train_model.py --model lstm` to bootstrap everything. From there you can branch into comparisons, more advanced architectures, or the visualization suite without relearning new entry points.
