# N-CMAPSS Engine RUL Prediction

Research-grade remaining useful life (RUL) prediction for NASA turbofan engines.

This repository is built around a single reproducible training pipeline for the N-CMAPSS dataset: one data loader, one CLI, one model registry, one metrics stack, and W&B logging for every serious run. The goal is simple: compare modern time-series architectures under controlled conditions and close the gap to published N-CMAPSS results.

## Current Results

| Claim | Model | Split / Protocol | RMSE | R2 | Accuracy@20 | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Best historical tuned run | CNN-GRU + asymmetric loss | FD1, tuned recipe | **6.44** | **0.91** | **99.1%** | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Best fair FD1 benchmark | WaveNet | 30 epochs, seq=1000, batch=32 | **6.523** | **0.9086** | **98.83%** | [FD1 report](benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/report.md) |
| Best fair FD2 benchmark | MSTCN | 30 epochs, seq=1000, batch=32 | **6.495** | **0.8897** | **99.01%** | [FD2 report](benchmark_results/apples_to_apples/fd2_ep30_len1000_20260425_112601/report.md) |
| Best FD2 threshold accuracy | WaveNet | 30 epochs, seq=1000, batch=32 | 6.739 | 0.8813 | **99.50%** | [FD2 report](benchmark_results/apples_to_apples/fd2_ep30_len1000_20260425_112601/report.md) |

The closest verified published reference we found reports **RMSE 6.20** on N-CMAPSS DS02. Under the repo's FD2 apples-to-apples check, **MSTCN is 0.295 RMSE away, or about 4.8%**. That is the most defensible current paper-gap number in this repository.

Paper reference: Jean-Pierre et al., PHM Society, "LSTM and Transformers based methods for Remaining Useful Life prediction considering censored data" ([PDF](https://papers.phmsociety.org/index.php/ijphm/article/download/4260/2619)).

W&B project for the current controlled benchmarks: [n-cmapss-a2a](https://wandb.ai/ptalur09-eastlake-high-school/n-cmapss-a2a)

## What This Repo Does

- Downloads and caches N-CMAPSS splits through `src/data/load_data.py`.
- Trains 20 registered neural architectures through `train_model.py`.
- Logs metrics, configs, plots, and benchmark summaries to Weights & Biases.
- Computes RUL-specific metrics: RMSE, MAE, R2, PHM score, asymmetric loss, Accuracy@10/15/20, and normalized RMSE.
- Provides strict apples-to-apples harnesses for comparing top models under the same split, training budget, loss, and metric denominator.

## Quick Start

```bash
pip install uv
uv sync

python train_model.py --list-models
python train_model.py --model mstcn --fd 1 --epochs 30 --max-seq-length 1000
```

Runs log to W&B online by default when you are logged in. Use `WANDB_MODE=offline` only for local-only experiments.

## Reproduce The Main Benchmarks

FD1 controlled benchmark:

```bash
python scripts/benchmark_apples_to_apples.py \
  --fd 1 \
  --epochs 30 \
  --max-sequence-length 1000 \
  --batch-size 32 \
  --patience-early-stop 6 \
  --patience-lr-reduce 3 \
  --fixed-metric-max-rul 125 \
  --models wavenet cnn_gru mstcn
```

FD2 / DS02-style controlled benchmark:

```bash
python scripts/benchmark_apples_to_apples.py \
  --fd 2 \
  --epochs 30 \
  --max-sequence-length 1000 \
  --batch-size 32 \
  --patience-early-stop 6 \
  --patience-lr-reduce 3 \
  --fixed-metric-max-rul 125 \
  --models wavenet cnn_gru mstcn
```

Each run writes `results.csv`, `results.json`, and `report.md` under `benchmark_results/apples_to_apples/`, and each model run is logged to W&B.

## Why The Apples-To-Apples Harness Matters

RUL papers often differ in dataset subset, censoring, windowing, target clipping, target scaling, sensor selection, normalization, and metric denominator. Small protocol differences can produce large apparent gains.

The apples-to-apples benchmark controls the parts we can control locally:

| Control | Current setting |
| --- | --- |
| Dataset selector | `--fd 1` or `--fd 2` |
| Epoch budget | `30` |
| Sequence length | `1000` |
| Batch size | `32` |
| Loss | `asymmetric_mse`, alpha `2.0` |
| Optimizer | Adam |
| Metric denominator | fixed max RUL `125` for normalized RMSE |
| Model set | WaveNet, CNN-GRU, MSTCN |

Use the fair benchmark tables for paper claims. Use historical tuned results for engineering direction and ablation discussion.

## Model Zoo

Run `python train_model.py --list-models` for the current registry.

| Family | Models |
| --- | --- |
| Convolutional / temporal CNN | `mstcn`, `atcn`, `cata_tcn`, `ttsnet`, `tcn`, `wavenet` |
| Attention | `transformer`, `attention_lstm`, `mdfa`, `cnn_lstm_attention` |
| Hybrid CNN-RNN | `cnn_gru`, `cnn_lstm`, `inception_lstm`, `resnet_lstm` |
| Recurrent | `lstm`, `bilstm`, `gru`, `bigru` |
| Baseline | `mlp` |

## Metrics

| Metric | Meaning |
| --- | --- |
| RMSE | Main error metric in cycles. Lower is better. |
| MAE | Mean absolute error in cycles. Lower is better. |
| R2 | Variance explained by the model. Higher is better. |
| PHM score | Safety-oriented score that penalizes late predictions more heavily. Lower is better. |
| Accuracy@N | Share of predictions within +/- N cycles. Higher is better. |
| RMSE(norm,fixed) | RMSE divided by a fixed denominator, currently `125`, for consistent internal paper-style comparison. |

Do not compare normalized RMSE numbers across papers unless the denominator and preprocessing are the same.

## W&B

The main benchmark project is:

```text
https://wandb.ai/ptalur09-eastlake-high-school/n-cmapss-a2a
```

Useful runs:

| Run | Model | Split | Link |
| --- | --- | --- | --- |
| `kxzt7837` | WaveNet | FD1 | https://wandb.ai/ptalur09-eastlake-high-school/n-cmapss-a2a/runs/kxzt7837 |
| `p3id5hq0` | MSTCN | FD2 | https://wandb.ai/ptalur09-eastlake-high-school/n-cmapss-a2a/runs/p3id5hq0 |
| `79nsc4w1` | WaveNet | FD2 | https://wandb.ai/ptalur09-eastlake-high-school/n-cmapss-a2a/runs/79nsc4w1 |

When reading W&B:

- Use `Summary` for final test metrics.
- Use `Charts` for convergence and overfitting.
- Use `Config` to verify whether two runs are actually comparable.
- Use benchmark summary runs and local reports for paper tables.

## Project Layout

```text
N-CMAPSS-Engine-Prediction
├── src/
│   ├── data/                # dataset download, cache, split loading
│   ├── models/              # model registry and architectures
│   └── utils/               # metrics and visualization helpers
├── train_model.py           # main training CLI
├── predict.py               # inference CLI and RULPredictor API
├── scripts/                 # benchmark, tuning, and reporting helpers
├── benchmark_results/       # reproducible benchmark outputs
├── pyproject.toml
└── uv.lock
```

## Tests

```bash
make test
pytest tests/test_metrics.py -v
```

The metric tests cover perfect predictions, known values, edge cases, PHM score asymmetry, and normalized metric scale behavior.

## Paper-Safe Summary

Use this wording when describing the current result:

> Under a controlled N-CMAPSS FD2 benchmark with 30 epochs, 1000-step windows, asymmetric MSE loss, and a fixed normalized-RMSE denominator of 125, MSTCN achieved RMSE 6.495, MAE 4.660, R2 0.890, and Accuracy@20 99.01%. This is within 0.295 RMSE, or 4.8%, of a verified published N-CMAPSS DS02 result of RMSE 6.20.

