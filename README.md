# N-CMAPSS Engine RUL Prediction

[![CI](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml)

Deep-learning benchmarks for remaining useful life (RUL) prediction on NASA's N-CMAPSS turbofan dataset. The repository contains a reproducible training pipeline, a shared model registry, controlled benchmark outputs, and a paper draft describing the results.

The main goal is not to present a single overfit leaderboard number. It is to compare recurrent, convolutional, transformer, WaveNet, and multi-scale temporal-attention models under the same preprocessing, loss, training budget, and metrics.

## Paper

- Draft PDF: [paper/main.pdf](paper/main.pdf)
- LaTeX source: [paper/main.tex](paper/main.tex)
- Canonical result map: [paper/state/canonical_results.md](paper/state/canonical_results.md)
- Review-response controlled suite: [benchmark_results/review_response/fd1_review_20260627_191300/report.md](benchmark_results/review_response/fd1_review_20260627_191300/report.md)

Build the paper with:

```bash
make -C paper
```

The paper intentionally separates supported findings from open ablations. In particular, sequence length is controlled only across 100-1000 timesteps, MSTCN component attribution is still open, and asymmetric loss is supported only by a limited WaveNet FD1 comparison.

## Controlled Results

These are the primary three-seed FD1 rows used for the final paper claims. They are simulation-benchmark results on N-CMAPSS FD1, not real-world deployment evidence or direct SOTA claims against papers using different protocols.

| Experiment | Model / Setting | Seeds | RMSE mean | RMSE std | R2 mean | Accuracy@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FD1 top cluster | WaveNet, T=1000 | 3 | **6.868** | 0.299 | **0.8986** | 98.24 |
| FD1 top cluster | MSTCN, T=1000 | 3 | 7.890 | 0.526 | 0.8659 | 97.75 |
| FD1 top cluster | CNN-GRU, T=1000 | 3 | 17.168 | 8.801 | 0.2561 | 64.13 |
| Window sweep | WaveNet, T=500 | 3 | **6.690** | 0.173 | **0.9038** | 98.24 |
| Window sweep | WaveNet, T=250 | 3 | 6.880 | 0.531 | 0.8980 | **98.44** |
| Loss comparison | WaveNet, asymmetric MSE | 3 | **6.868** | 0.299 | **0.8986** | **98.24** |
| Loss comparison | WaveNet, MSE | 3 | 7.214 | 0.559 | 0.8878 | 97.65 |

Reports:

- [Review-response FD1 controlled suite](benchmark_results/review_response/fd1_review_20260627_191300/report.md)
- [SOTA-chase best-seed sweep](benchmark_results/sota_chase/fd1_review_20260628_120711/report.md)
- [Earlier FD1 controlled benchmark](benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/report.md)
- [Earlier FD2 controlled benchmark](benchmark_results/apples_to_apples/fd2_ep30_len1000_20260425_112601/report.md)

## Best Single Run

For leaderboard-style comparison only, the best individual FD1 run found so far is:

| Model | Setting | Seed | RMSE | R2 | Accuracy@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| WaveNet | T=1000, asymmetric MSE, 32 features | 47 | **6.197** | **0.9175** | **99.71** |

This is intentionally reported separately from the controlled three-seed table because it is a best-seed result, not a robust mean/std claim.

## Main Findings

- Short operational windows are effective on N-CMAPSS FD1. In a matched WaveNet sweep, T=500 is best by mean RMSE, with T=250 and T=1000 close behind.
- WaveNet is the strongest stable FD1 model in the three-seed suite. MSTCN is stable and competitive, but not dominant. CNN-GRU is unstable under these settings.
- Asymmetric MSE is modestly better than symmetric MSE in the WaveNet FD1 loss comparison, but broader alpha and architecture sweeps remain future work.

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
  --reader-max-rul 65 \
  --fixed-metric-max-rul 65 \
  --models wavenet cnn_gru mstcn
```

Run the review-response suite:

```bash
WANDB_MODE=offline uv run python scripts/run_review_response_experiments.py
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
