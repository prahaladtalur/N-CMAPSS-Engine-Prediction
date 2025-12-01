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
