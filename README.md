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
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/               # Model architectures & training
│   │   ├── lstm_model.py
│   │   ├── rnn_model.py
│   │   └── train.py
│   │
│   ├── utils/                # Utilities
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── config.py
│   │
│   └── evaluate.py           # Evaluation script
│
├── main.py                   # Main entry point to run pipeline
├── requirements.txt          # Dependencies
├── uv.lock                   # uv dependency lockfile
├── pyproject.toml            # uv project config
├── README.md                 # Project documentation
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
python main.py
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

### Loss Functions for RUL Prediction

The project includes **6 specialized loss functions** that emphasize accuracy where it matters most:

- **combined_rul** ⭐ (default): Balances critical zone focus + asymmetric penalty
- **phm_score**: Official PHM competition scoring
- **weighted_mse**: Focus on critical zone (RUL < 30)
- **asymmetric_mse**: Penalize under-prediction more (safety first)
- **quantile_90**: Conservative estimates
- **mse**: Standard baseline

```bash
# Use custom loss function
python train_model.py --model transformer --loss combined_rul

# Compare different loss functions
python train_model.py --compare --models lstm gru --loss phm_score
```

**Why custom loss?** Standard MSE treats a 10-cycle error at RUL=200 the same as at RUL=20 (critical!). Our custom losses emphasize:
1. Critical zone performance (low RUL values matter more)
2. Asymmetric costs (under-prediction = catastrophic, over-prediction = just expensive)

For detailed loss function documentation, see [LOSS_FUNCTIONS.md](LOSS_FUNCTIONS.md).

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
