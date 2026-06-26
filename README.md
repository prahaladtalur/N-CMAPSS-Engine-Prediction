# N-CMAPSS Engine RUL Prediction

[![CI](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml)

Deep-learning benchmarks for remaining useful life (RUL) prediction on NASA's N-CMAPSS turbofan dataset. The repository contains a reproducible training pipeline, a shared model registry, controlled benchmark outputs, and a paper draft describing the results.

The main goal is not to present a single overfit leaderboard number. It is to compare recurrent, convolutional, transformer, WaveNet, and multi-scale temporal-attention models under the same preprocessing, loss, training budget, and metrics.

## Paper

- Draft PDF: [paper/main.pdf](paper/main.pdf)
- LaTeX source: [paper/main.tex](paper/main.tex)
- Canonical result map: [paper/state/canonical_results.md](paper/state/canonical_results.md)

Build the paper with:

```bash
make -C paper
```

The paper intentionally separates supported findings from open ablations. In particular, sequence length and architecture are not fully disentangled, and asymmetric loss is framed as a motivated design choice rather than a proven empirical improvement.

## Controlled Results

These are the primary apples-to-apples rows used for paper claims.

| Split | Model | Protocol | RMSE | R2 | PHM | Accuracy@20 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FD1 | WaveNet | 30 epochs, T=1000, batch=32 | **6.523** | **0.9086** | **0.7256** | **98.83** |
| FD1 | CNN-GRU | 30 epochs, T=1000, batch=32 | 7.006 | 0.8946 | 0.8396 | 98.53 |
| FD1 | MSTCN | 30 epochs, T=1000, batch=32 | 7.604 | 0.8758 | 1.0033 | 98.53 |
| FD2 | MSTCN | 30 epochs, T=1000, batch=32 | **6.495** | **0.8897** | **0.5755** | 99.01 |
| FD2 | WaveNet | 30 epochs, T=1000, batch=32 | 6.739 | 0.8813 | 0.6756 | **99.50** |
| FD2 | CNN-GRU | 30 epochs, T=1000, batch=32 | 19.812 | -0.0265 | 5.1463 | 59.41 |

Reports:

- [FD1 controlled benchmark](benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/report.md)
- [FD2 controlled benchmark](benchmark_results/apples_to_apples/fd2_ep30_len1000_20260425_112601/report.md)

## Main Findings

- Short operational windows around 1000 timesteps are effective on N-CMAPSS, while full-flight recurrent baselines perform much worse. This is a directional finding because the current full-length and short-window rows also differ by architecture family.
- MSTCN is in the top performance cluster, ranks third on controlled FD1, and is best by RMSE on FD2. The current evidence supports competitiveness, not universal dominance.
- Asymmetric MSE is used because late RUL predictions are operationally more dangerous than early predictions. A symmetric-loss baseline and alpha sweep remain future work.

## Installation

This project uses `uv` for reproducible Python environments.

```bash
git clone https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction.git
cd N-CMAPSS-Engine-Prediction

pip install uv
uv sync --all-extras
```

The N-CMAPSS data is downloaded through `rul-datasets` when training scripts run. Raw data, trained models, local W&B runs, and generated result directories are intentionally ignored by Git.

## Quick Start

List the registered models:

```bash
uv run python train_model.py --list-models
```

Train one model on FD1 with the standard short-window setup:

```bash
uv run python train_model.py \
  --model mstcn \
  --fd 1 \
  --epochs 30 \
  --batch-size 32 \
  --max-seq-length 1000
```

Run the controlled FD1 benchmark used in the paper:

```bash
uv run python scripts/benchmark_apples_to_apples.py \
  --fd 1 \
  --epochs 30 \
  --max-sequence-length 1000 \
  --batch-size 32 \
  --patience-early-stop 6 \
  --patience-lr-reduce 3 \
  --fixed-metric-max-rul 125 \
  --models wavenet cnn_gru mstcn
```

## Model Registry

Run `uv run python train_model.py --list-models` for the authoritative list.

| Family | Models |
| --- | --- |
| Convolutional / temporal CNN | `mstcn`, `atcn`, `cata_tcn`, `ttsnet`, `tcn`, `wavenet` |
| Attention / transformer | `transformer`, `attention_lstm`, `mdfa`, `cnn_lstm_attention`, `star_transformer`, `sparse_transformer_bigrcu` |
| Hybrid CNN-RNN | `cnn_gru`, `cnn_lstm`, `inception_lstm`, `resnet_lstm` |
| Recurrent | `lstm`, `bilstm`, `gru`, `bigru` |
| Baseline | `mlp` |

## Repository Layout

```text
.
|-- src/                      # data loading, model registry, metrics, visualization
|-- tests/                    # unit tests for models, metrics, prediction, training helpers
|-- scripts/                  # benchmark, tuning, comparison, and reporting scripts
|-- benchmark_results/        # committed benchmark summaries used by the paper
|-- paper/                    # LaTeX manuscript, figures, references, and evidence map
|-- docs/                     # supporting documentation and archived historical reports
|-- train_model.py            # main training CLI
|-- predict.py                # inference CLI and RULPredictor API
|-- pyproject.toml
`-- uv.lock
```

## Tests And Checks

```bash
uv run black --check src/ tests/ train_model.py scripts/
uv run mypy src/ train_model.py
WANDB_MODE=offline uv run pytest tests/ -v --tb=short
```

For a faster smoke check:

```bash
WANDB_MODE=offline uv run pytest tests/test_metrics.py tests/test_models.py -q
```

## Notes For Reviewers

- Historical reports were moved to [docs/archive](docs/archive) because several were written before the final paper synthesis and contain stronger claims than the controlled evidence supports.
- The committed benchmark outputs are small summaries. Large generated artifacts, model checkpoints, local logs, `.env`, W&B runs, and raw N-CMAPSS data are not committed.
- This repository is research code, not an aviation-certified maintenance system.

## Citation

If this repository is useful, cite it with the metadata in [CITATION.cff](CITATION.cff).
