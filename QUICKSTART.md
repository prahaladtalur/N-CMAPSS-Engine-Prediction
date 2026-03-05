# Quick Start Guide
## Train Your First RUL Prediction Model in 5 Minutes

This guide gets you from zero to trained model in the fastest way possible.

---

## 1. Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction.git
cd N-CMAPSS-Engine-Prediction

# Install dependencies
pip install uv          # Install uv package manager
uv sync                 # Install all project dependencies

# Verify installation
python train_model.py --list-models
```

**Expected output**: List of 20 available models

---

## 2. Train the Best Model (3 minutes)

```bash
# Train MSTCN (winner of our 20-model benchmark)
WANDB_MODE=offline python train_model.py \
  --model mstcn \
  --epochs 30 \
  --batch-size 64 \
  --max-seq-length 1000 \
  --fd 1

# Takes ~3 minutes on modern hardware
```

**What happens**:
1. ✅ Downloads N-CMAPSS FD1 dataset (~100MB, cached for future runs)
2. ✅ Trains MSTCN model for 30 epochs with early stopping
3. ✅ Generates training curves, prediction plots, error analysis
4. ✅ Logs metrics to W&B (offline mode)

**Expected results**:
- RMSE: ~6.8-7.5 cycles
- R²: ~0.88-0.90
- Accuracy@20: >98%

---

## 3. View Results

After training completes, check:

### Training Logs
```bash
# See test set metrics
grep "RMSE:" wandb/latest-run/logs/debug.log
```

### Visualizations
```bash
# Results saved to:
ls results/mstcn-run/
# Output:
# - training_history.png      (loss curves)
# - predictions.png            (true vs predicted scatter plot)
# - error_distribution.png     (histogram of errors)
```

### W&B Dashboard
```bash
# View detailed metrics
wandb offline sync wandb/latest-run/

# Or just explore the local files
ls wandb/offline-run-*/
```

---

## 4. Compare Multiple Models (Optional)

```bash
# Compare top 3 models
WANDB_MODE=offline python train_model.py \
  --compare \
  --models mstcn transformer wavenet \
  --epochs 30 \
  --batch-size 64 \
  --max-seq-length 1000 \
  --fd 1

# Takes ~9 minutes (3 min × 3 models)
```

**Results**:
- Comparison plot: `results/comparison/model_comparison.png`
- Individual metrics for each model
- Best model automatically identified

---

## 5. Production Deployment (Optional)

Once you have a trained model, deploy it:

```python
# Save your model
from keras.models import load_model

# Load from W&B run directory
model = load_model('wandb/offline-run-*/files/model.h5')

# Make predictions
import numpy as np
test_sequence = np.random.randn(1, 1000, 32)  # Your sensor data
rul_prediction = model.predict(test_sequence)
print(f"Predicted RUL: {rul_prediction[0][0]:.2f} cycles")
```

See [README_PRODUCTION.md](README_PRODUCTION.md) for full deployment guide.

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Solution: Run uv sync
uv sync
```

### Issue: "Out of Memory"
```bash
# Solution: Reduce batch size
python train_model.py --model mstcn --batch-size 32 --max-seq-length 1000
```

### Issue: "W&B API key error"
```bash
# Solution: Use offline mode
WANDB_MODE=offline python train_model.py --model mstcn
```

### Issue: Training very slow
```bash
# Check: Are you using short sequences?
# ✅ Correct:   --max-seq-length 1000
# ❌ Incorrect: No flag (uses full 20,294 timesteps = very slow!)
```

---

## Next Steps

### Learn More
- [README.md](README.md) - Full documentation
- [MSTCN_EXPLAINED.md](MSTCN_EXPLAINED.md) - How the best model works
- [FINAL_ANALYSIS_REPORT.md](FINAL_ANALYSIS_REPORT.md) - Complete benchmark
- [CLAUDE.md](CLAUDE.md) - Developer guide

### Explore Models
```bash
# List all 20 models
python train_model.py --list-models

# Get recommendations
python train_model.py --recommend

# Train different architectures
python train_model.py --model transformer
python train_model.py --model wavenet
python train_model.py --model atcn
```

### Try Different Datasets
```bash
# N-CMAPSS has 7 subsets (FD1-FD7)
python train_model.py --model mstcn --fd 2
python train_model.py --model mstcn --fd 3
# ... etc
```

### Hyperparameter Tuning
```bash
# Experiment with different settings
python train_model.py --model mstcn \
  --epochs 50 \
  --units 128 \
  --dropout 0.3 \
  --lr 0.0005
```

### Contribute
Found a better architecture? Improved results? Open a PR!

1. Create an issue describing your improvement
2. Create a feature branch
3. Add your model to `src/models/architectures.py`
4. Run `make check` before committing
5. Open a PR with results

---

## Performance Expectations

Based on our comprehensive 20-model benchmark:

| Model | RMSE | R² | Training Time | When to Use |
|-------|------|-----|---------------|-------------|
| **MSTCN** | **6.80** | **0.90** | 3 min | **Production (best overall)** |
| Transformer | 6.82 | 0.90 | 3 min | Edge devices (fewer params) |
| WaveNet | 6.84 | 0.90 | 3 min | Speed-critical apps |
| ATCN | 7.01 | 0.89 | 3 min | Good alternative |
| TCN | 16.13 | 0.44 | 25 min* | Baseline comparison |
| LSTM | 22.28 | -0.07 | 30 min* | ❌ Don't use |

*Without `--max-seq-length` flag (uses full sequences)

---

## Key Takeaways

✅ **Use MSTCN** for best results (RMSE 6.80, R² 0.90)
✅ **Always set** `--max-seq-length 1000` (58% better performance!)
✅ **30 epochs** is sufficient (early stopping around epoch 25-35)
✅ **Batch size 64** works well for most systems
✅ **Use offline W&B** if you don't have an API key

❌ **Avoid traditional RNNs** (LSTM/GRU) - they have negative R² scores
❌ **Don't use full sequences** (20K+ timesteps) - very slow, poor results
❌ **Don't train 100+ epochs** - marginal gains, 3x longer

---

## Getting Help

- **Documentation**: Start with [README.md](README.md)
- **Issues**: https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/issues
- **Model details**: See [MSTCN_EXPLAINED.md](MSTCN_EXPLAINED.md)

---

**Happy modeling!** 🚀

*Last updated: March 4, 2026*
