# Jet Engine Remaining Useful Life Prediction

This project uses NASA's **C-MAPSS turbofan engine dataset** to build predictive models (LSTM/RNN) that estimate the **Remaining Useful Life (RUL)** of jet engines. The goal is to enable predictive maintenance by identifying how many flight cycles remain before an engine requires servicing, based on multivariate time-series sensor data.

## Dataset
- **Primary Dataset**: [NASA C-MAPSS Turbofan Engine Data](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Alternate sources:
  - [Kaggle Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
  - [PHM Society Archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)

The dataset contains simulated sensor readings from jet engines under different operating conditions and fault scenarios. The task is framed as a **time-series regression problem**, where the model predicts RUL from sensor degradation patterns.

## Project Structure
```
N-CMAPSS-Engine-Prediction/
│
├── data/                     # Datasets
│   ├── raw/                  # Original raw data
│   ├── processed/            # Preprocessed/cleaned data
│   └── external/             # Additional/reference data
│
├── src/                      # Source code
│   ├── data/                 # Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── models/               # Model architectures & training
│   │   ├── __init__.py
│   │   ├── architectures.py  # All 13 model architectures
│   │   ├── lstm_model.py     # Legacy LSTM model
│   │   └── train.py          # Training pipeline
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── metrics.py        # Evaluation metrics
│       ├── visualize.py      # Data & model visualizations
│       └── training_viz.py   # Training visualizations
│
├── scripts/                  # Example scripts
│   └── example_visualizations.py  # Visualization examples
│
├── notebooks/                # Jupyter notebooks
│   └── N_CMAPSS.ipynb
│
├── train_model.py            # Main training CLI
├── requirements.txt          # Dependencies
├── uv.lock                   # uv dependency lockfile
├── pyproject.toml            # uv project config
├── README.md                 # Project documentation
├── MODEL_SELECTION.md        # Model selection guide
├── VISUALIZATIONS.md         # Visualization guide
└── .gitignore                # Ignore datasets, logs, etc.
```

## Setup
This project uses [**uv**](https://github.com/astral-sh/uv) for environment and dependency management (instead of pip or conda).

### Installation
```bash
# Install uv if not already installed
pip install uv

# Sync environment
uv sync
```

### Running the project
```bash
# Train a model using the CLI
python train_model.py --model lstm
```

## Model Architectures

This project includes **13 state-of-the-art models** for RUL prediction with easy switching:

### Available Models:
- **RNN-based:** LSTM, BiLSTM, GRU, BiGRU, Attention-LSTM, ResNet-LSTM
- **Convolutional:** TCN, WaveNet
- **Hybrid:** CNN-LSTM, CNN-GRU, Inception-LSTM
- **Attention:** Transformer
- **Baseline:** MLP

### Quick Start:

```bash
# List all available models
python train_model.py --list-models

# Train a single model
python train_model.py --model lstm

# Compare multiple models
python train_model.py --compare --models lstm gru transformer

# Get model recommendations
python train_model.py --recommend
```

For detailed model selection guidance, see [MODEL_SELECTION.md](MODEL_SELECTION.md).

## Visualization Capabilities

This project includes comprehensive visualization tools for both data analysis and model evaluation:

### Data Analysis Visualizations
- **Sensor Degradation Analysis** - Visualize how sensors change as engines degrade
- **Sensor Correlation Heatmap** - Identify which sensors are most predictive of failure
- **Multi-Sensor Lifecycle Comparison** - Compare sensor behaviors side-by-side

### Model Evaluation Visualizations
- **RUL Trajectory Analysis** - Track predicted vs actual RUL over engine lifecycle
- **Critical Zone Analysis** - Evaluate performance when engines are near failure
- **Prediction Confidence** - Visualize uncertainty and confidence intervals
- **Training History** - Loss and metrics over epochs
- **Error Distribution** - Analyze prediction errors

### Quick Start with Visualizations

```bash
# Run interactive visualization examples
python example_visualizations.py

# Or import specific visualizations in your code
from src.utils.visualize import (
    plot_sensor_degradation,
    plot_critical_zone_analysis,
    plot_rul_trajectory
)
```

For detailed documentation on all visualization functions, see [VISUALIZATIONS.md](VISUALIZATIONS.md).

## Next Steps
- Implement preprocessing pipeline (scaling, windowing, sequence creation).
- Develop LSTM and RNN architectures.
- Train models and evaluate using RMSE and other scoring functions.

---
