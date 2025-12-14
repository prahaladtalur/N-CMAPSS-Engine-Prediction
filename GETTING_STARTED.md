# Complete Getting Started Guide

This guide walks you through everything from installation to training models and running visualizations.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Download Dataset](#download-dataset)
4. [Quick Start - Train Your First Model](#quick-start---train-your-first-model)
5. [Explore Available Models](#explore-available-models)
6. [Train Different Models](#train-different-models)
7. [Compare Multiple Models](#compare-multiple-models)
8. [Visualize Data and Results](#visualize-data-and-results)
9. [Complete Workflow Example](#complete-workflow-example)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **pip** (for installing uv)
- **Internet connection** (for downloading dataset)

---

## Installation

### Step 1: Install uv (Package Manager)

```bash
# Install uv using pip
pip install uv
```

**Note:** If you prefer not to use `uv`, you can use `pip` directly, but `uv` is faster and recommended.

### Step 2: Clone or Navigate to Project

```bash
# If you have the project in a repository
git clone <your-repo-url>
cd N-CMAPSS-Engine-Prediction

# Or if you already have the project
cd /path/to/N-CMAPSS-Engine-Prediction
```

### Step 3: Install Dependencies

```bash
# Using uv (recommended - creates virtual environment automatically)
uv sync

# OR using pip (if you prefer)
pip install -r requirements.txt
```

This installs all required packages:
- TensorFlow (for deep learning models)
- NumPy, Pandas (data processing)
- Matplotlib, Seaborn (visualizations)
- rul-datasets (for automatic dataset download)
- Wandb (for experiment tracking)
- And more...

**✅ You're now ready to start!**

---

## Download Dataset

The dataset is **automatically downloaded** when you first run the training script. No manual download needed!

However, if you want to test data loading separately:

```bash
# Test data loading (downloads dataset automatically)
python -c "from src.data.load_data import get_datasets; get_datasets(fd=1)"
```

The dataset will be cached in `data/raw/` directory. The first run downloads ~50MB of data.

**Available Datasets:**
- FD001 (default) - Training dataset
- FD002 - Test dataset with 6 operating conditions
- FD003 - Test dataset with 1 fault mode
- FD004 - Test dataset with 6 operating conditions + 1 fault mode
- FD005, FD006, FD007 - Additional test scenarios

---

## Quick Start - Train Your First Model

### Option 1: Train with Default Settings (Easiest)

```bash
# Train a simple LSTM model with default settings
python train_model.py --model lstm
```

This will:
1. ✅ Automatically download the dataset (if not already downloaded)
2. ✅ Load and preprocess the data
3. ✅ Train an LSTM model for 50 epochs
4. ✅ Evaluate on test set
5. ✅ Save results to `results/` directory
6. ✅ Log metrics to Wandb (if configured)

### Option 2: Quick Training (Fewer Epochs)

```bash
# Train quickly with fewer epochs (good for testing)
python train_model.py --model gru --epochs 10
```

**Expected Output:**
```
Loading N-CMAPSS FD1 dataset...
✓ N-CMAPSS FD1 prepared and cached in: data/raw
✓ Data loaded:
  - Dev units: 100
  - Test units: 100

Training: gru
Building gru model...
Model Architecture:
...
Starting training...
Epoch 1/10
...
Training Complete!
Test Set Metrics:
  RMSE: 25.34
  MAE: 18.92
  ...
```

---

## Explore Available Models

### List All Models

```bash
# See all 13 available models
python train_model.py --list-models
```

**Output:**
```
AVAILABLE MODELS FOR RUL PREDICTION
================================================================================

RNN-based Models:
  • lstm                  - Standard LSTM - baseline RNN for sequence modeling
  • bilstm                - Bidirectional LSTM - captures both past and future context
  • gru                   - GRU - simpler than LSTM, often faster with similar performance
  ...
```

### Get Model Recommendations

```bash
# Get recommendations based on your needs
python train_model.py --recommend
```

**Output:**
```
MODEL RECOMMENDATIONS
================================================================================

Quick Baseline (fast experiments):
  Recommended: mlp, gru

Best Accuracy (maximum performance):
  Recommended: transformer, attention_lstm, wavenet, resnet_lstm

Fastest Training (quick iterations):
  Recommended: gru, cnn_gru, tcn
...
```

---

## Train Different Models

### Train a Single Model

```bash
# Train LSTM
python train_model.py --model lstm

# Train Transformer (SOTA model)
python train_model.py --model transformer --epochs 100 --units 128

# Train TCN (fast and accurate)
python train_model.py --model tcn --epochs 50 --units 64

# Train Attention LSTM (interpretable)
python train_model.py --model attention_lstm --epochs 80
```

### Customize Training Parameters

```bash
# Full customization example
python train_model.py \
  --model transformer \
  --fd 1 \
  --epochs 100 \
  --units 128 \
  --dense-units 64 \
  --dropout 0.3 \
  --lr 0.0005 \
  --batch-size 64 \
  --max-seq-length 50
```

**Parameter Guide:**
- `--model`: Model architecture (lstm, gru, transformer, etc.)
- `--fd`: Dataset index (1-7, default: 1)
- `--epochs`: Training epochs (default: 50)
- `--units`: Hidden units in recurrent layers (default: 64)
- `--dense-units`: Dense layer units (default: 32)
- `--dropout`: Dropout rate (default: 0.2)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--max-seq-length`: Max sequence length (default: None = full sequences)

### Train on Different Datasets

```bash
# Train on FD001 (default)
python train_model.py --model lstm --fd 1

# Train on FD002 (more complex)
python train_model.py --model lstm --fd 2

# Train on FD003
python train_model.py --model lstm --fd 3
```

---

## Compare Multiple Models

### Compare Specific Models

```bash
# Compare 3 models
python train_model.py --compare --models lstm gru transformer

# Compare RNN variants
python train_model.py --compare --models lstm bilstm gru bigru
```

### Compare ALL Models (Takes Time!)

```bash
# Compare all 13 models (this will take a while!)
python train_model.py --compare-all --epochs 30
```

**Note:** This trains all 13 models sequentially. Use `--epochs 30` for faster comparison.

### Compare with Custom Settings

```bash
# Compare with custom configuration
python train_model.py \
  --compare \
  --models transformer attention_lstm wavenet \
  --epochs 100 \
  --units 128 \
  --batch-size 64
```

Results are saved to:
- `results/comparison/model_comparison.png` - Visual comparison
- Wandb project: `n-cmapss-rul-comparison`

---

## Visualize Data and Results

### Run Visualization Examples

```bash
# Interactive visualization menu
python scripts/example_visualizations.py

# Or run specific visualizations
python scripts/example_visualizations.py --data --fd 1
python scripts/example_visualizations.py --model --fd 1
python scripts/example_visualizations.py --all --fd 1
```

### Use Visualizations in Your Code

```python
from src.data.load_data import get_datasets
from src.utils import (
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_critical_zone_analysis,
    plot_rul_trajectory,
)

# Load data
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

# Analyze sensor degradation
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)

# See sensor correlations
plot_sensor_correlation_heatmap(dev_X, dev_y)

# After training, analyze model performance
# (assuming you have y_true and y_pred)
plot_critical_zone_analysis(y_true, y_pred)
plot_rul_trajectory(y_true, y_pred, unit_length=[len(y) for y in test_y], unit_idx=0)
```

For complete visualization documentation, see [VISUALIZATIONS.md](VISUALIZATIONS.md).

---

## Complete Workflow Example

Here's a complete workflow from start to finish:

### Step 1: Setup (One-time)

```bash
# Install dependencies
uv sync
```

### Step 2: Explore Models

```bash
# See what models are available
python train_model.py --list-models

# Get recommendations
python train_model.py --recommend
```

### Step 3: Quick Test Run

```bash
# Quick test with GRU (fast training)
python train_model.py --model gru --epochs 10
```

### Step 4: Train Best Models

```bash
# Train top recommended models
python train_model.py --model transformer --epochs 100 --units 128
python train_model.py --model attention_lstm --epochs 100
python train_model.py --model wavenet --epochs 100
```

### Step 5: Compare Models

```bash
# Compare your trained models
python train_model.py --compare --models transformer attention_lstm wavenet --epochs 50
```

### Step 6: Visualize Results

```bash
# Run visualization examples
python scripts/example_visualizations.py --all
```

### Step 7: Analyze Best Model

```python
# In Python, load your best model and analyze
from src.models.train import train_model
from src.data.load_data import get_datasets
from src.utils import plot_critical_zone_analysis

# Load data
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)
val_X, val_y = val if val else (None, None)

# Train and get predictions
model, history, metrics = train_model(
    dev_X=dev_X,
    dev_y=dev_y,
    model_name="transformer",
    test_X=test_X,
    test_y=test_y,
    config={"epochs": 100, "units": 128}
)

# Analyze critical zone performance
# (You'll need to get y_pred from model.predict())
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
# Make sure you activated the virtual environment
# With uv:
source .venv/bin/activate  # On Mac/Linux
# OR
.venv\Scripts\activate  # On Windows

# Then verify installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Issue: "Dataset download fails"

**Solution:**
```bash
# Check internet connection
# Try manual download from:
# https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

# Or use Kaggle:
# https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
```

### Issue: "Out of memory during training"

**Solution:**
```bash
# Reduce batch size
python train_model.py --model lstm --batch-size 16

# Reduce model size
python train_model.py --model lstm --units 32

# Limit sequence length
python train_model.py --model lstm --max-seq-length 30
```

### Issue: "Training is too slow"

**Solution:**
```bash
# Use faster models
python train_model.py --model gru  # Faster than LSTM
python train_model.py --model tcn  # Very fast

# Reduce epochs for testing
python train_model.py --model lstm --epochs 10

# Increase batch size (if memory allows)
python train_model.py --model lstm --batch-size 64
```

### Issue: "Wandb authentication error"

**Solution:**
```bash
# Option 1: Login to Wandb
wandb login

# Option 2: Disable Wandb (training still works)
# Wandb is optional - training will work without it
```

### Issue: "CUDA/GPU not found"

**Solution:**
- TensorFlow will automatically use CPU if GPU is not available
- Training will work, just slower
- To use GPU, install TensorFlow with GPU support:
  ```bash
  pip install tensorflow[and-cuda]
  ```

---

## Next Steps

1. **Read Model Selection Guide**: [MODEL_SELECTION.md](MODEL_SELECTION.md)
2. **Explore Visualizations**: [VISUALIZATIONS.md](VISUALIZATIONS.md)
3. **Experiment with Hyperparameters**: Try different `--units`, `--lr`, `--dropout` values
4. **Try Different Datasets**: Test on FD002, FD003, etc.
5. **Compare Models**: Find the best model for your use case

---

## Quick Reference

### Most Common Commands

```bash
# List models
python train_model.py --list-models

# Train single model
python train_model.py --model lstm

# Compare models
python train_model.py --compare --models lstm gru transformer

# Quick test
python train_model.py --model gru --epochs 10

# Visualize
python scripts/example_visualizations.py
```

### Best Models by Use Case

- **Quick Baseline**: `gru`, `mlp`
- **Best Accuracy**: `transformer`, `attention_lstm`, `wavenet`
- **Fast Training**: `gru`, `tcn`, `cnn_gru`
- **Interpretable**: `lstm`, `attention_lstm`
- **Long Sequences**: `tcn`, `wavenet`, `transformer`

---

**Happy Training! 🚀**

For questions or issues, check the main [README.md](README.md) or open an issue.

