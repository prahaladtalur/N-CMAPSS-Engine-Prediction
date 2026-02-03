# Production Model - Quick Start Guide

## What You Get

A production-ready CNN-GRU model that predicts Remaining Useful Life (RUL) of turbofan engines with **99.1% accuracy** (within ±20 cycles).

**Model Performance:**
- RMSE: 6.44 cycles
- R² Score: 0.91
- Accuracy@20: 99.1%
- PHM Score: 0.73

## Three Simple Steps

### 1️⃣ Train the Model

```bash
WANDB_MODE=offline python train_production_model.py
```

This takes ~30-40 minutes and saves the model to `models/production/cnn_gru_best.keras`.

### 2️⃣ Run Predictions

```bash
# Evaluate on test set
python predict.py --model-path models/production/cnn_gru_best.keras

# Predict specific engine
python predict.py --model-path models/production/cnn_gru_best.keras --unit-id 5

# Custom data
python predict.py --model-path models/production/cnn_gru_best.keras --input-file data.npy
```

### 3️⃣ See Examples

```bash
python example_usage.py
```

Shows 5 real-world examples:
- Single engine prediction
- Fleet monitoring (batch prediction)
- Maintenance scheduling
- Confidence assessment
- Custom sensor data

## Python API

```python
from predict import RULPredictor

# Load model
predictor = RULPredictor("models/production/cnn_gru_best.keras")

# Predict RUL (sensor_data: shape = (timesteps, 20))
rul = predictor.predict_single(sensor_data)

if rul < 20:
    print(f"CRITICAL: {rul:.1f} cycles remaining - schedule maintenance NOW")
```

## Files

| File | Purpose |
|------|---------|
| `train_production_model.py` | Train and save the model |
| `predict.py` | Inference script with CLI and Python API |
| `example_usage.py` | 5 real-world usage examples |
| `DEPLOYMENT.md` | Full deployment documentation |
| `EXPERIMENTS.md` | Research results and architecture comparison |

## Key Features

✅ **Best Architecture**: CNN-GRU beats Transformer by 4.6% RMSE
✅ **Asymmetric Loss**: Penalizes dangerous late predictions 2× more
✅ **Safety-First**: 99.1% predictions within ±20 cycles
✅ **Production-Ready**: Saved model, inference API, examples
✅ **Battle-Tested**: Trained on NASA N-CMAPSS dataset

## Next Steps

- **Deploy as API**: Wrap `RULPredictor` in Flask/FastAPI
- **Real-time Monitoring**: Stream sensor data → predict RUL → alert
- **Fine-tune**: Retrain on your domain-specific engine data
- **Ensemble**: Combine CNN-GRU + Transformer + WaveNet for extra accuracy

## Need Help?

- Full docs: [DEPLOYMENT.md](DEPLOYMENT.md)
- Research: [EXPERIMENTS.md](EXPERIMENTS.md)
- Code: [train_model.py](train_model.py)
