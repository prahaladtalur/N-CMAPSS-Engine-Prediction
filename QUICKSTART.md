# Quick Start

This is the shortest path from a clean clone to a reproducible N-CMAPSS training run.

## 1. Install

```bash
git clone https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction.git
cd N-CMAPSS-Engine-Prediction

pip install uv
uv sync --all-extras
```

## 2. Verify The Model Registry

```bash
uv run python train_model.py --list-models
```

## 3. Train A Single Model

```bash
WANDB_MODE=offline uv run python train_model.py \
  --model mstcn \
  --fd 1 \
  --epochs 30 \
  --batch-size 32 \
  --max-seq-length 1000
```

## 4. Reproduce The Controlled Benchmark

```bash
WANDB_MODE=offline uv run python scripts/benchmark_apples_to_apples.py \
  --fd 1 \
  --epochs 30 \
  --max-sequence-length 1000 \
  --batch-size 32 \
  --patience-early-stop 6 \
  --patience-lr-reduce 3 \
  --reader-max-rul 65 \
  --fixed-metric-max-rul 65 \
  --models wavenet cnn_gru mstcn
```

Outputs are written under `benchmark_results/apples_to_apples/` and include `results.csv`, `results.json`, and `report.md`.

## 5. Build The Paper

```bash
make -C paper
```

The generated manuscript is `paper/main.pdf`.

## Useful Checks

```bash
uv run black --check src/ tests/ train_model.py scripts/
uv run mypy src/ train_model.py
WANDB_MODE=offline uv run pytest tests/test_metrics.py tests/test_models.py -q
```
