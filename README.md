# Jet Engine Remaining Useful Life Prediction

This project uses NASA's **N-CMAPSS turbofan engine dataset** to build predictive models that estimate the **Remaining Useful Life (RUL)** of jet engines. The goal is to enable predictive maintenance by identifying how many flight cycles remain before an engine requires servicing, based on multivariate time-series sensor data.

## Dataset

**Dataset**: NASA N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation)
- Automatically downloaded via `rul-datasets` library
- 7 sub-datasets (FD001-FD007) with different operating conditions and fault modes
- Multivariate time-series sensor data with Remaining Useful Life (RUL) labels

The dataset contains simulated sensor readings from jet engines under different operating conditions and fault scenarios. The task is framed as a **time-series regression problem**, where the model predicts RUL from sensor degradation patterns.

## Project Structure

```
N-CMAPSS-Engine-Prediction/
│
├── train.py                  # Main training entry point
├── visualize.py              # Data visualization entry point
│
├── src/                      # Source code
│   ├── data/                 # Data loading
│   │   ├── __init__.py
│   │   └── loader.py         # Dataset loading utilities
│   │
│   ├── models/               # Model architectures & training
│   │   ├── __init__.py
│   │   ├── architectures.py  # 13 SOTA model architectures
│   │   └── trainer.py        # Training pipeline
│   │
│   ├── evaluation/           # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py        # RUL-specific metrics
│   │
│   └── visualization/        # Visualization utilities
│       ├── __init__.py
│       ├── data_viz.py       # Data exploration visualizations
│       └── model_viz.py      # Model evaluation visualizations
│
├── outputs/                  # All outputs (gitignored)
│   ├── models/               # Saved models
│   ├── figures/              # Generated visualizations
│   └── logs/                 # Training logs
│
├── data/                     # Dataset (auto-downloaded)
│   ├── raw/                  # Original N-CMAPSS data
│   └── processed/            # Processed/cached data
│
├── notebooks/                # Jupyter notebooks
│   └── N_CMAPSS.ipynb
│
├── requirements.txt          # Dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use uv for faster installation
pip install uv
uv sync
```

### Train a Model

```bash
# Train a single model
python train.py --model lstm

# Train with custom configuration
python train.py --model transformer --epochs 100 --units 128

# Compare multiple models
python train.py --compare --models lstm gru attention_lstm

# Compare all available models
python train.py --compare-all

# List available models
python train.py --list-models

# Get model recommendations
python train.py --recommend
```

### Visualize Data

```bash
# Visualize dataset (basic)
python visualize.py --data --fd 1

# Show all visualizations
python visualize.py --data --all --fd 1

# Visualize specific sensors
python visualize.py --data --fd 1 --sensors 0 1 2 3

# Visualize specific unit
python visualize.py --data --fd 1 --unit 5
```

## Model Architectures

This project includes **13 state-of-the-art models** for RUL prediction with easy switching:

### Available Models:
- **RNN-based:** LSTM, BiLSTM, GRU, BiGRU, Attention-LSTM, ResNet-LSTM
- **Convolutional:** TCN, WaveNet
- **Hybrid:** CNN-LSTM, CNN-GRU, Inception-LSTM
- **Attention:** Transformer
- **Baseline:** MLP

### Model Recommendations:

**For Best Accuracy:**
- Attention-LSTM, Transformer, Inception-LSTM

**For Fast Training:**
- MLP, GRU, LSTM

**For Long Sequences:**
- TCN, WaveNet, Transformer

**For Limited Data:**
- LSTM, GRU, BiGRU

For detailed model selection guidance, see [MODEL_SELECTION.md](MODEL_SELECTION.md).

## Visualization Capabilities

### Data Exploration Visualizations
- **RUL Distribution** - Understand label distribution across datasets
- **Sensor Time Series** - Visualize sensor readings over time
- **Sensor Degradation** - Analyze how sensors change as engines degrade
- **Sensor Correlation** - Identify which sensors are most predictive
- **Multi-Sensor Lifecycle** - Compare sensor behaviors side-by-side

### Model Evaluation Visualizations
- **Training History** - Loss and metrics over epochs
- **Predictions vs Actual** - Scatter plots and residual analysis
- **Error Distribution** - Analyze prediction errors
- **RUL Trajectory** - Track predicted vs actual RUL over lifecycle
- **Critical Zone Analysis** - Performance when engines are near failure
- **Model Comparison** - Compare multiple models side-by-side

For detailed documentation on all visualization functions, see [VISUALIZATIONS.md](VISUALIZATIONS.md).

## Usage Examples

### Training Example

```python
from src.data.loader import get_datasets
from src.models.trainer import train_model

# Load data
(dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=1)
val_X, val_y = val_pair if val_pair else (None, None)

# Train model
model, history, metrics = train_model(
    dev_X=dev_X,
    dev_y=dev_y,
    model_name="attention_lstm",
    val_X=val_X,
    val_y=val_y,
    test_X=test_X,
    test_y=test_y,
    config={"epochs": 50, "batch_size": 32}
)
```

### Visualization Example

```python
from src.data.loader import get_datasets
from src.visualization.data_viz import (
    plot_rul_distribution,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap
)

# Load data
(dev_X, dev_y), _, (test_X, test_y) = get_datasets(fd=1)

# Visualize data
plot_rul_distribution(dev_y, split_name="Development Set")
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)
plot_sensor_correlation_heatmap(dev_X, dev_y)
```

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

- **Standard Metrics:** MSE, RMSE, MAE, MAPE, R²
- **RUL-Specific:** PHM Score (asymmetric penalty for late predictions)
- **Accuracy Metrics:** Percentage within ±10, ±15, ±20 cycles

## Project Features

✅ **13 State-of-the-Art Models** - Easy switching between architectures
✅ **Comprehensive Visualizations** - Data exploration and model evaluation
✅ **Wandb Integration** - Automatic experiment tracking
✅ **Clean CLI Interface** - Simple commands for training and visualization
✅ **Modular Architecture** - Well-organized, maintainable codebase
✅ **RUL-Specific Metrics** - Industry-standard evaluation metrics

## Contributing

Feel free to open issues or submit pull requests for improvements!

## License

MIT License

---

**Happy Predicting! 🚀**
