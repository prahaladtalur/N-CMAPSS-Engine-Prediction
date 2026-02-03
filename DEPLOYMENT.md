# Production Deployment Guide

## CNN-GRU RUL Predictor

This guide shows how to use the production-ready CNN-GRU model for Remaining Useful Life (RUL) prediction.

## Quick Start

### 1. Train Production Model

```bash
# Train and save the best model (CNN-GRU with asymmetric loss)
WANDB_MODE=offline python train_production_model.py

# Custom output directory
python train_production_model.py --output-dir models/my_model
```

This will:
- Train CNN-GRU with optimal hyperparameters (from experiments)
- Save model to `models/production/cnn_gru_best.keras`
- Save config to `models/production/config.json`
- Save metrics to `models/production/metrics.json`

### 2. Make Predictions

#### Evaluate on Test Set

```bash
# Full test set evaluation with visualizations
python predict.py --model-path models/production/cnn_gru_best.keras

# Different dataset (FD1-FD7)
python predict.py --model-path models/production/cnn_gru_best.keras --fd 2
```

#### Predict Specific Unit

```bash
# Predict RUL for test unit #5
python predict.py --model-path models/production/cnn_gru_best.keras --unit-id 5
```

Output:
```
Unit 5:
  True RUL:      42.50 cycles
  Predicted RUL: 44.23 cycles
  Error:         1.73 cycles
```

#### Custom Input Data

```bash
# Predict from custom .npy file
python predict.py --model-path models/production/cnn_gru_best.keras --input-file data.npy
```

Input format: NumPy array with shape `(timesteps, features)` where features = 20 sensor readings.

## Python API

### Basic Usage

```python
from predict import RULPredictor
import numpy as np

# Initialize predictor
predictor = RULPredictor(model_path="models/production/cnn_gru_best.keras")

# Load your sensor data (timesteps × 20 features)
sensor_data = np.load("engine_sensor_data.npy")

# Predict RUL
rul = predictor.predict_single(sensor_data)
print(f"Predicted RUL: {rul:.2f} cycles")
```

### Batch Predictions

```python
# Predict multiple units
units = [unit1_data, unit2_data, unit3_data]  # List of arrays
predictions = predictor.predict_batch(units)
```

### Full Evaluation

```python
# Evaluate on test set with metrics
metrics = predictor.evaluate_test_set(
    fd=1,
    visualize=True,
    output_dir="results/my_predictions"
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")
print(f"Accuracy@20: {metrics['accuracy_20']:.2f}%")
```

## Model Architecture

**CNN-GRU Hybrid**
- Conv1D (64 filters, kernel=3) → MaxPool
- Conv1D (32 filters, kernel=3) → MaxPool
- GRU (64 units, return sequences)
- Dense (32 units, ReLU)
- Dense (1 unit, linear) → RUL prediction

**Training Details**
- Loss: Asymmetric MSE (α=2.0, penalizes late predictions 2×)
- Optimizer: Adam (lr=0.001)
- Dropout: 0.2
- Batch size: 32
- Sequence length: 1000 timesteps (truncated from 20,294)
- Early stopping: patience=20
- Parameters: ~70k

## Performance Benchmarks

Trained on N-CMAPSS FD1 (30 epochs typical convergence):

| Metric | Value |
|--------|-------|
| **RMSE** | **6.44** |
| **MAE** | **4.87** |
| **PHM Score** | **0.73** |
| **R² Score** | **0.91** |
| **Acc@10** | **90.0%** |
| **Acc@20** | **99.1%** |

**Accuracy@20 = 99.1%** means the model predicts within ±20 cycles on 99% of test samples.

## Production Checklist

- [x] Model trained with optimal hyperparameters
- [x] Asymmetric loss for safety-critical predictions
- [x] Early stopping to prevent overfitting
- [x] Model saved in Keras format (.keras)
- [x] Config and metrics saved for reproducibility
- [x] Inference API with single/batch prediction
- [x] Full test set evaluation
- [x] Visualization of predictions and errors

## Next Steps

1. **Deploy to API**: Wrap `RULPredictor` in Flask/FastAPI endpoint
2. **Real-time Monitoring**: Stream sensor data → predict RUL → trigger alerts
3. **Model Versioning**: Use MLflow or similar for model registry
4. **A/B Testing**: Compare CNN-GRU vs ensemble (Transformer + WaveNet + CNN-GRU)
5. **Fine-tuning**: Retrain on domain-specific data if available

## Troubleshooting

**Q: Model fails to load with loss function error**
A: The `RULPredictor` class automatically loads the asymmetric MSE loss. Make sure you're using the `predict.py` script or the `RULPredictor` class.

**Q: Predictions seem off**
A: Check input data format (timesteps × 20 features) and ensure sensor readings are normalized using the same scaler as training data.

**Q: Memory issues with long sequences**
A: Sequences are automatically truncated to 1000 timesteps (configurable via `max_sequence_length` in config).

## References

- **Experiments**: See [EXPERIMENTS.md](EXPERIMENTS.md) for full architecture comparison
- **Training Code**: [train_production_model.py](train_production_model.py)
- **Inference Code**: [predict.py](predict.py)
- **Dataset**: NASA N-CMAPSS (rul_datasets library)
