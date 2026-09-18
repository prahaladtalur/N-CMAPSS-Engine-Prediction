# N-CMAPSS Engine Maintenance-Warning Study

[![CI](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml)

This repository supports the paper **“Beyond Cycle Count: Sensor-Based Maintenance Warnings for Simulated Turbofan Engines.”** The current study asks whether measured telemetry improves final-window warning performance beyond elapsed cycle and recorded operating conditions on the public, 100-times-downsampled tiny-N-CMAPSS simulation subset.

Under nested engine-held-out evaluation with thresholds selected inside the training folds, adding 14 measured physical channels raises final-window recall from **55.4% to 79.3%** at an approximately 5% pre-window false-alert rate. The paired complete-engine bootstrap interval for the 23.9-percentage-point gain is 18.3 to 29.6 points. These are simulation results, not a service interval or fielded-aircraft decision rule.

## Current paper and reproducibility materials

- Revised manuscript: [paper/ijscar_submission/main_ijscar.pdf](paper/ijscar_submission/main_ijscar.pdf)
- LaTeX entry point: [paper/ijscar_submission/main_ijscar.tex](paper/ijscar_submission/main_ijscar.tex)
- Corrected observable-input results: [benchmark_results/maintenance_warning/corrected_observable_nested_20260812/results.json](benchmark_results/maintenance_warning/corrected_observable_nested_20260812/results.json)
- Nested model-selection results: [benchmark_results/maintenance_warning/model_selected_observable_20260814_robust/results.json](benchmark_results/maintenance_warning/model_selected_observable_20260814_robust/results.json)
- Primary evaluation runner: [scripts/run_observable_maintenance_warning_evaluation.py](scripts/run_observable_maintenance_warning_evaluation.py)
- Nested model-selection runner: [scripts/run_nested_model_selection_warning_evaluation.py](scripts/run_nested_model_selection_warning_evaluation.py)
- Dataset source: [tiny-N-CMAPSS](https://github.com/alovberg/tiny-N-CMAPSS) at pinned commit [`b915997ff8d571f4e9d091954d6556835f212ded`](https://github.com/alovberg/tiny-N-CMAPSS/tree/b915997ff8d571f4e9d091954d6556835f212ded)

The committed result bundles contain source-file SHA256 hashes, input definitions, outer-test predictions, selected thresholds, fold metrics, first-warning records, model configurations, bootstrap intervals, and seed-robustness summaries. The models exclude `Fc`, `hs`, unit ID, and RUL from their inputs.

## Earlier architecture-comparison materials

The repository also retains the earlier RUL-regression benchmarks for historical and engineering context. They are not the evidence base for the current maintenance-warning paper.

- Draft PDF: [output/pdf/n-cmapss-rul-paper-draft.pdf](output/pdf/n-cmapss-rul-paper-draft.pdf)
- LaTeX source: [paper/main.tex](paper/main.tex)
- Canonical result map: [paper/state/canonical_results.md](paper/state/canonical_results.md)
- Three-seed controlled suite: [benchmark_results/review_response/fd1_review_20260627_191300/report.md](benchmark_results/review_response/fd1_review_20260627_191300/report.md)

Build the paper with:

```bash
make -C paper
```

The paper intentionally separates supported findings from open ablations. In particular, sequence length is controlled only across 100-1000 timesteps, MSTCN component attribution is still open, and asymmetric loss is supported only by a limited WaveNet FD1 comparison.

## Legacy controlled RUL-regression results

These are the primary three-seed FD1 rows from the earlier architecture-comparison draft. They are simulation-benchmark results on N-CMAPSS FD1, not evidence for the current maintenance-warning claim, real-world deployment, or direct state-of-the-art comparisons against papers using different protocols.

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

- [Three-seed FD1 controlled suite](benchmark_results/review_response/fd1_review_20260627_191300/report.md)
- [Targeted best-seed sweep](benchmark_results/sota_chase/fd1_review_20260628_120711/report.md)
- [Earlier FD1 controlled benchmark](benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/report.md)
- [Earlier FD2 controlled benchmark](benchmark_results/apples_to_apples/fd2_ep30_len1000_20260425_112601/report.md)

## Legacy best single run

For leaderboard-style comparison only, the best individual FD1 run found so far is:

| Model | Setting | Seed | RMSE | R2 | Accuracy@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| WaveNet | T=1000, asymmetric MSE, 32 features | 47 | **6.197** | **0.9175** | **99.71** |

This is intentionally reported separately from the controlled three-seed table because it is a best-seed result, not a robust mean/std claim.

## Legacy benchmark findings

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

The N-CMAPSS data is downloaded through `rul-datasets` when training scripts run. Raw data, trained models, and local W&B runs are intentionally ignored by Git. Compact result bundles used by the current paper are committed for auditability.

## Quick Start

List the registered models:

```bash
uv run python train_model.py --list-models
```

Train one model on FD1 with the standard short-window setup:

```bash
WANDB_MODE=offline uv run python train_model.py \
  --model mstcn \
  --fd 1 \
  --epochs 30 \
  --batch-size 32 \
  --max-seq-length 1000
```

Run the controlled FD1 benchmark used in the paper:

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

Run the three-seed controlled suite:

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
