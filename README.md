# Beyond Cycle Count

[![CI](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/actions/workflows/ci.yml)

Code and reproducibility record for **“Beyond Cycle Count: Sensor-Based Maintenance Warnings for Simulated Turbofan Engines,”** accepted by the *International Journal of Scientific Computing and Artificial Research* (IJSCAR).

The study asks whether measured telemetry identifies cycles inside a simulated engine's final 20-cycle window better than elapsed cycle and operating conditions alone. Under nested engine-held-out evaluation, adding summaries from 14 measured channels raises final-window recall from **55.4% to 79.3%** at an approximately 5% pre-window false-alert rate. The paired complete-engine bootstrap interval for the 23.9-percentage-point gain is 18.3 to 29.6 points.

These are results on the public, 100-times-downsampled tiny-N-CMAPSS simulation subset. They do not define a service interval or validate a fielded-aircraft maintenance rule.

## Accepted paper and exact release

- [Accepted manuscript PDF](paper/ijscar_submission/main_ijscar.pdf)
- [LaTeX entry point](paper/ijscar_submission/main_ijscar.tex)
- [Release `v1.0-ijscar`](https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/releases/tag/v1.0-ijscar)
- Dataset: [tiny-N-CMAPSS](https://github.com/alovberg/tiny-N-CMAPSS) at pinned commit [`b915997ff8d571f4e9d091954d6556835f212ded`](https://github.com/alovberg/tiny-N-CMAPSS/tree/b915997ff8d571f4e9d091954d6556835f212ded)

For the exact publication state, check out the release tag instead of `main`:

```bash
git clone --branch v1.0-ijscar \
  https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction.git
cd N-CMAPSS-Engine-Prediction
```

## Reproducibility map

| Publication evidence | Location |
| --- | --- |
| Four-input comparison, thresholds, complete-engine bootstrap intervals, fixed-threshold check, warning timing, and five-seed results | [`benchmark_results/maintenance_warning/corrected_observable_nested_20260812/`](benchmark_results/maintenance_warning/corrected_observable_nested_20260812/) |
| Nested model-family selection, outer-test predictions, selected models, warning timing, and five-seed results | [`benchmark_results/maintenance_warning/model_selected_observable_20260814_robust/`](benchmark_results/maintenance_warning/model_selected_observable_20260814_robust/) |
| Primary four-input evaluation | [`scripts/run_observable_maintenance_warning_evaluation.py`](scripts/run_observable_maintenance_warning_evaluation.py) |
| Nested model-selection evaluation | [`scripts/run_nested_model_selection_warning_evaluation.py`](scripts/run_nested_model_selection_warning_evaluation.py) |
| Tables 1–6 rendered from the committed JSON | [`scripts/render_ijscar_tables.py`](scripts/render_ijscar_tables.py) |

The result bundles record the source-file SHA256 hashes, feature definitions, selected thresholds, model configurations, fold metrics, outer-test predictions, complete-engine bootstrap intervals, first-warning records, and seed-robustness summaries. Every model excludes `Fc`, `hs`, unit ID, and RUL from its inputs.

## Reproduce the reported tables

Install the locked environment with [`uv`](https://docs.astral.sh/uv/):

```bash
pip install uv
uv sync --all-extras
```

To render Tables 1–6 directly from the committed publication results:

```bash
uv run python scripts/render_ijscar_tables.py \
  --output reproduced_tables.md
```

To rerun the underlying experiments from the pinned, checksum-verified data and create fresh result bundles:

```bash
uv run python scripts/run_observable_maintenance_warning_evaluation.py \
  --data-dir data/tiny_ncmapss \
  --output-dir reproduced_results/observable \
  --download

uv run python scripts/run_nested_model_selection_warning_evaluation.py \
  --data-dir data/tiny_ncmapss \
  --output-dir reproduced_results/nested_selection \
  --download
```

Then render the tables from the new outputs:

```bash
uv run python scripts/render_ijscar_tables.py \
  --primary-results reproduced_results/observable/results.json \
  --selected-results reproduced_results/nested_selection/results.json \
  --output reproduced_tables.md
```

The default settings reproduce the paper protocol: a 20-cycle final window, a 5% pre-window false-alert budget, five outer engine-held-out folds, four inner engine-grouped folds, 2,000 complete-engine bootstrap resamples, and seeds 2024–2028. The complete runs may take time; the committed JSON and CSV files permit immediate auditing without retraining.

## Build the accepted manuscript

The accepted manuscript uses Tectonic:

```bash
cd paper/ijscar_submission
tectonic main_ijscar.tex
```

## Repository layout

```text
.
|-- benchmark_results/maintenance_warning/  # publication result bundles
|-- paper/ijscar_submission/                 # accepted PDF and LaTeX entry point
|-- paper/sections/maintenance_*.tex         # accepted study sections
|-- scripts/run_*warning_evaluation.py       # evaluation runners
|-- scripts/render_ijscar_tables.py          # JSON-to-table renderer
|-- src/                                     # shared model and data utilities
|-- tests/                                   # unit tests
|-- pyproject.toml
`-- uv.lock
```

## Earlier work

This repository began as an RUL architecture-benchmark project. That work is retained for historical context, but it is not the evidence base for the accepted maintenance-warning paper.

- [Earlier RUL-regression draft](output/pdf/n-cmapss-rul-paper-draft.pdf)
- [Earlier LaTeX source](paper/main.tex)
- [Three-seed FD1 architecture comparison](benchmark_results/review_response/fd1_review_20260627_191300/report.md)
- [Earlier best-seed sweep](benchmark_results/sota_chase/fd1_review_20260628_120711/report.md)
- [Archived reports](docs/archive/)

The earlier WaveNet/MSTCN/CNN-GRU tables and best-seed leaderboard are historical experiments. They should not be read as results from the accepted warning study.

## Tests and checks

```bash
uv run black --check src/ tests/ train_model.py scripts/
uv run mypy src/ train_model.py
WANDB_MODE=offline uv run pytest tests/ -v --tb=short
```

This repository contains research code for a simulation study, not an aviation-certified maintenance system.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).
