# Complete Guide to N-CMAPSS Engine RUL Prediction

This guide covers **everything** you need to know to use this project effectively.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Models](#training-models)
4. [Visualizing Data](#visualizing-data)
5. [Using the Python API](#using-the-python-api)
6. [Understanding the Project Structure](#understanding-the-project-structure)
7. [Available Models](#available-models)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Install Dependencies

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using uv (faster)
pip install uv
uv sync
```

### Step 2: Verify Installation

```bash
python train.py --list-models
```

You should see a list of 13 available models.

---

## Quick Start

### Train Your First Model (30 seconds)

```bash
# Train a simple LSTM model on dataset FD1
python train.py --model lstm --fd 1 --epochs 10
```

This will:
1. Download the N-CMAPSS FD1 dataset automatically
2. Train an LSTM model for 10 epochs
3. Generate visualizations in `outputs/figures/lstm-run/`
4. Log results to Weights & Biases (if configured)

### Visualize the Data

```bash
# See basic data visualizations
python visualize.py --data --fd 1

# See all available visualizations
python visualize.py --data --all --fd 1
```

---

## Training Models

### Basic Training

```bash
# Train a specific model
python train.py --model lstm --fd 1

# Available models: lstm, gru, bilstm, bigru, attention_lstm,
#                   transformer, tcn, wavenet, cnn_lstm, cnn_gru,
#                   inception_lstm, resnet_lstm, mlp
```

### Custom Training Configuration

```bash
# Train with custom hyperparameters
python train.py --model transformer \
  --fd 1 \
  --epochs 100 \
  --batch-size 64 \
  --units 128 \
  --dense-units 64 \
  --dropout 0.3 \
  --lr 0.0001
```

### Compare Multiple Models

```bash
# Compare specific models
python train.py --compare --models lstm gru attention_lstm --fd 1

# Compare ALL 13 models (takes a while!)
python train.py --compare-all --fd 1
```

Comparison results saved to: `outputs/figures/comparison/model_comparison.png`

### Training Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model architecture to train | Required |
| `--fd` | Dataset index (1-7) | 1 |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Batch size | 32 |
| `--units` | Number of hidden units | 64 |
| `--dense-units` | Dense layer units | 32 |
| `--dropout` | Dropout rate | 0.2 |
| `--lr` | Learning rate | 0.001 |
| `--max-seq-length` | Max sequence length | None (use full) |
| `--project` | Wandb project name | n-cmapss-rul-prediction |
| `--run-name` | Custom run name | model-name-run |
| `--no-normalize` | Disable feature normalization | False |
| `--no-visualize` | Disable visualizations | False |

### Get Model Recommendations

```bash
# See which models work best for different scenarios
python train.py --recommend
```

This shows recommendations for:
- Best accuracy
- Fastest training
- Long sequences
- Limited data
- Complex patterns
- Most interpretable

---

## Visualizing Data

### Basic Visualizations

```bash
# Visualize dataset FD1
python visualize.py --data --fd 1
```

This shows:
- RUL distribution (development and test sets)
- Sensor time series for Unit 0

### All Visualizations

```bash
# Show everything!
python visualize.py --data --all --fd 1
```

This includes:
- RUL distributions
- Sensor time series
- Sensor degradation patterns
- Sensor correlation heatmap
- Multi-sensor lifecycle comparison

### Custom Visualizations

```bash
# Visualize specific sensors
python visualize.py --data --fd 1 --sensors 0 1 2 3

# Visualize a different unit
python visualize.py --data --fd 1 --unit 5

# Combine options
python visualize.py --data --all --fd 1 --unit 3 --sensors 0 1 2 3 4 5
```

### Visualization Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Enable data visualization mode | Required |
| `--fd` | Dataset index (1-7) | 1 |
| `--unit` | Unit index to visualize | 0 |
| `--sensors` | Sensor indices (space-separated) | 0 1 2 3 |
| `--all` | Show all visualizations | False |

---

## Using the Python API

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
    config={
        "epochs": 50,
        "batch_size": 32,
        "units": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.001
    },
    project_name="my-project",
    run_name="my-experiment",
    normalize=True,
    visualize=True
)

# Access results
print(f"Test RMSE: {metrics['rmse']:.2f}")
print(f"Test MAE: {metrics['mae']:.2f}")
print(f"PHM Score: {metrics['phm_score']:.2f}")
```

### Data Visualization Example

```python
from src.data.loader import get_datasets
from src.visualization.data_viz import (
    plot_rul_distribution,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_multi_sensor_lifecycle
)

# Load data
(dev_X, dev_y), _, (test_X, test_y) = get_datasets(fd=1)

# Create visualizations
plot_rul_distribution(dev_y, split_name="Development Set")
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)
plot_sensor_correlation_heatmap(dev_X, dev_y)
plot_multi_sensor_lifecycle(dev_X, dev_y, unit_idx=0, max_sensors=8)
```

### Model Evaluation Example

```python
from src.evaluation.metrics import compute_all_metrics, format_metrics
import numpy as np

# Assume you have predictions
y_true = np.array([100, 90, 80, 70, 60])
y_pred = np.array([95, 88, 82, 72, 58])

# Compute all metrics
metrics = compute_all_metrics(y_true, y_pred)

# Print formatted results
print(format_metrics(metrics))

# Access specific metrics
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"PHM Score: {metrics['phm_score']:.2f}")
print(f"Accuracy@10: {metrics['accuracy_10']:.1f}%")
```

### Model Prediction Visualization Example

```python
from src.visualization.model_viz import (
    plot_predictions,
    plot_error_distribution,
    plot_rul_trajectory,
    plot_critical_zone_analysis
)

# Visualize predictions
plot_predictions(y_true, y_pred, model_name="LSTM")
plot_error_distribution(y_true, y_pred, model_name="LSTM")
plot_rul_trajectory(y_true, y_pred)
plot_critical_zone_analysis(y_true, y_pred)
```

---

## Understanding the Project Structure

```
N-CMAPSS-Engine-Prediction/
│
├── train.py                    # Main training CLI
├── visualize.py                # Visualization CLI
│
├── src/                        # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Dataset loading
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py    # 13 model definitions
│   │   └── trainer.py          # Training pipeline
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Evaluation metrics
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── data_viz.py         # Data visualizations
│       └── model_viz.py        # Model visualizations
│
├── outputs/                    # All outputs (gitignored)
│   ├── models/                 # Saved models
│   ├── figures/                # Visualizations
│   └── logs/                   # Training logs
│
├── data/                       # Dataset (auto-downloaded)
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── N_CMAPSS.ipynb          # Example notebook
│
└── [config files]
```

### What Each File Does

**Entry Points:**
- `train.py` - Train models, compare models, get recommendations
- `visualize.py` - Visualize dataset and predictions

**Core Modules:**
- `src/data/loader.py` - Downloads and loads N-CMAPSS data
- `src/models/architectures.py` - Defines all 13 model architectures
- `src/models/trainer.py` - Training pipeline with wandb integration
- `src/evaluation/metrics.py` - RUL-specific evaluation metrics
- `src/visualization/data_viz.py` - Dataset exploration plots
- `src/visualization/model_viz.py` - Model evaluation plots

---

## Available Models

### RNN-Based Models

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| `lstm` | Standard LSTM | General purpose, baseline | ⚡⚡⚡ |
| `bilstm` | Bidirectional LSTM | Better accuracy, small datasets | ⚡⚡ |
| `gru` | Gated Recurrent Unit | Faster than LSTM, similar accuracy | ⚡⚡⚡⚡ |
| `bigru` | Bidirectional GRU | Best of both worlds | ⚡⚡ |
| `attention_lstm` | LSTM + Attention | **Best accuracy**, interpretable | ⚡ |
| `resnet_lstm` | LSTM + Residual connections | Deep networks, long sequences | ⚡⚡ |

### Convolutional Models

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| `tcn` | Temporal Convolutional Network | Long sequences, parallel training | ⚡⚡⚡ |
| `wavenet` | Dilated convolutions | Very long sequences | ⚡⚡ |

### Hybrid Models

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| `cnn_lstm` | CNN feature extraction + LSTM | Multi-scale patterns | ⚡⚡ |
| `cnn_gru` | CNN feature extraction + GRU | Faster than CNN-LSTM | ⚡⚡⚡ |
| `inception_lstm` | Multi-scale CNN + LSTM | Complex patterns | ⚡ |

### Attention-Based Models

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| `transformer` | Self-attention mechanism | **State-of-the-art**, long sequences | ⚡ |

### Baseline

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| `mlp` | Multi-layer perceptron | Quick experiments, baseline | ⚡⚡⚡⚡⚡ |

### Model Selection Guide

**For Best Accuracy:**
1. `attention_lstm` - LSTM with attention mechanism
2. `transformer` - State-of-the-art self-attention
3. `inception_lstm` - Multi-scale feature extraction

**For Fast Training:**
1. `mlp` - Fastest, no temporal modeling
2. `gru` - Fast and effective
3. `lstm` - Good balance

**For Long Sequences:**
1. `tcn` - Temporal Convolutional Network
2. `wavenet` - Dilated convolutions
3. `transformer` - Self-attention

**For Limited Data:**
1. `lstm` - Simple, less prone to overfit
2. `gru` - Even simpler
3. `bigru` - More capacity than GRU

---

## Evaluation Metrics

### Standard Regression Metrics

- **MSE** (Mean Squared Error) - Standard loss metric
- **RMSE** (Root Mean Squared Error) - In same units as RUL
- **MAE** (Mean Absolute Error) - Average absolute error
- **MAPE** (Mean Absolute Percentage Error) - Percentage error
- **R²** (R-squared) - Variance explained

### RUL-Specific Metrics

- **PHM Score** - Official PHM08 Challenge scoring function
  - Penalizes late predictions (predicting failure after actual) more heavily
  - Lower is better, 0 is perfect

- **Asymmetric Loss** - Custom loss that penalizes late predictions 2x more

### Accuracy Metrics

- **Accuracy@10** - Percentage of predictions within ±10 cycles
- **Accuracy@15** - Percentage of predictions within ±15 cycles
- **Accuracy@20** - Percentage of predictions within ±20 cycles

### Understanding Metrics

```python
from src.evaluation.metrics import compute_all_metrics, format_metrics

metrics = compute_all_metrics(y_true, y_pred)
print(format_metrics(metrics))

# Output:
# Evaluation Metrics:
# ----------------------------------------
#   MSE:              123.4567
#   RMSE:             11.1111
#   MAE:              8.8888
#   MAPE:             12.34%
#   R2 Score:         0.8765
# ----------------------------------------
#   PHM Score:        1234.56
#   PHM Score (norm): 12.3456
#   Asymmetric Loss:  234.5678
# ----------------------------------------
#   Accuracy@10:      45.67%
#   Accuracy@15:      67.89%
#   Accuracy@20:      78.90%
```

---

## Advanced Usage

### Custom Model Configuration

```python
config = {
    "units": 128,                    # Hidden units
    "dense_units": 64,               # Dense layer units
    "dropout_rate": 0.3,             # Dropout rate
    "learning_rate": 0.0001,         # Learning rate
    "batch_size": 64,                # Batch size
    "epochs": 100,                   # Number of epochs
    "max_sequence_length": 50,       # Limit sequence length
    "patience_early_stop": 10,       # Early stopping patience
    "patience_lr_reduce": 5,         # LR reduction patience
}

model, history, metrics = train_model(
    dev_X=dev_X,
    dev_y=dev_y,
    model_name="transformer",
    config=config
)
```

### Comparing Models Programmatically

```python
from src.models.trainer import compare_models

results = compare_models(
    dev_X=dev_X,
    dev_y=dev_y,
    model_names=["lstm", "gru", "attention_lstm"],
    test_X=test_X,
    test_y=test_y,
    config={"epochs": 30}
)

# Access results
for model_name, result in results.items():
    if "metrics" in result:
        print(f"{model_name}: RMSE = {result['metrics']['rmse']:.2f}")
```

### Custom Visualizations

```python
from src.visualization.model_viz import create_evaluation_report

# Generate comprehensive report
create_evaluation_report(
    model_name="LSTM",
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    history=history,
    save_dir="outputs/figures/my-experiment"
)
```

### Working with Different Datasets

```python
# Load different FD datasets
for fd in range(1, 8):
    print(f"\nTraining on FD{fd}...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=fd)

    model, history, metrics = train_model(
        dev_X=dev_X,
        dev_y=dev_y,
        model_name="lstm",
        test_X=test_X,
        test_y=test_y,
        config={"epochs": 10},
        run_name=f"lstm-fd{fd}"
    )

    print(f"FD{fd} RMSE: {metrics['rmse']:.2f}")
```

---

## Troubleshooting

### Common Issues

**Problem: ModuleNotFoundError**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem: CUDA out of memory**
```bash
# Solution: Reduce batch size
python train.py --model lstm --batch-size 16
```

**Problem: Training is too slow**
```bash
# Solution: Use a faster model or reduce epochs
python train.py --model gru --epochs 20
```

**Problem: Dataset download fails**
```bash
# Solution: Check internet connection and retry
# The rul-datasets library will auto-download on first use
```

**Problem: Visualizations not showing**
```bash
# Solution: Make sure matplotlib backend is configured
# For headless systems, visualizations are saved to outputs/figures/
```

### Getting Help

1. Check the README.md
2. Look at example code in notebooks/N_CMAPSS.ipynb
3. Use `--help` flag: `python train.py --help`
4. Open an issue on GitHub

---

## Quick Reference

### Most Common Commands

```bash
# List available models
python train.py --list-models

# Train LSTM on FD1
python train.py --model lstm --fd 1

# Compare top 3 models
python train.py --compare --models attention_lstm transformer inception_lstm

# Visualize data
python visualize.py --data --all --fd 1

# Get model recommendations
python train.py --recommend
```

### Most Useful Python Imports

```python
# Data
from src.data.loader import get_datasets

# Training
from src.models.trainer import train_model, compare_models

# Metrics
from src.evaluation.metrics import compute_all_metrics, format_metrics

# Visualizations
from src.visualization.data_viz import (
    plot_rul_distribution,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap
)
from src.visualization.model_viz import (
    plot_predictions,
    plot_error_distribution,
    plot_rul_trajectory,
    plot_critical_zone_analysis
)
```

---

## That's It! 🚀

You now know everything you need to use this project effectively. Start with the Quick Start section and explore from there!

**Happy Predicting!**
