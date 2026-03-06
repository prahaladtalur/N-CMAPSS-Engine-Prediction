# Automatic Model Checkpointing

## Overview

As of March 5, 2026, `train_model.py` now automatically saves complete model checkpoints after every training run. This eliminates the need to retrain models for inference and enables reproducible predictions.

## What Gets Saved

After each training run, a timestamped directory is created with:

```
models/<model_name>_<timestamp>/
├── model.keras              # Trained Keras model
├── scaler.pkl              # Fitted StandardScaler for feature normalization
├── config.json             # Training configuration (hyperparameters)
├── metrics.json            # Test set evaluation metrics
├── rul_scaler.json         # RUL normalization parameters (y_min, y_max)
├── metadata.json           # Complete model metadata
└── README.md               # Usage instructions
```

## Example Usage

### Training with Auto-Save (Default)

```bash
# Train MSTCN - checkpoint saved automatically
python train_model.py --model mstcn --epochs 30 --max-seq-length 1000

# Output:
# ============================================================
# Saving model checkpoint to: models/mstcn_20260305_143022
# ============================================================
# ✓ Model saved: model.keras
# ✓ Scaler saved: scaler.pkl
# ✓ RUL scaler saved: rul_scaler.json
# ✓ Config saved: config.json
# ✓ Metrics saved: metrics.json
# ✓ Metadata saved: metadata.json
# ✓ README saved: README.md
# ============================================================
# ✅ Model checkpoint saved successfully!
# 📂 Location: models/mstcn_20260305_143022
# ============================================================
```

### Disable Auto-Save

```bash
# Skip checkpoint saving (for quick experiments)
python train_model.py --model mstcn --no-save
```

### Compare Mode

```bash
# Compare 3 models - each gets its own checkpoint
python train_model.py --compare --models mstcn transformer wavenet

# Creates:
# - models/mstcn_20260305_143022/
# - models/transformer_20260305_143524/
# - models/wavenet_20260305_144012/
```

## Loading Saved Models

### Using predict.py

```bash
# Single model prediction
python predict.py --model-path models/mstcn_20260305_143022/model.keras --fd 1

# Predict on specific unit
python predict.py --model-path models/mstcn_20260305_143022/model.keras --unit-id 5

# Custom data file
python predict.py --model-path models/mstcn_20260305_143022/model.keras --input-file data.npy
```

### Using Python API

```python
from predict import RULPredictor
import numpy as np

# Load predictor with saved checkpoint
predictor = RULPredictor(
    model_path='models/mstcn_20260305_143022/model.keras'
)

# Make prediction
sensor_data = np.load('sensor_readings.npy')  # Shape: (timesteps, 32)
result = predictor.predict_single(sensor_data)

print(f"Predicted RUL: {result['prediction']:.2f} cycles")
```

### Manual Loading (Advanced)

```python
import json
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

checkpoint_dir = 'models/mstcn_20260305_143022'

# 1. Load model
model = load_model(f'{checkpoint_dir}/model.keras')

# 2. Load scaler
with open(f'{checkpoint_dir}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 3. Load config
with open(f'{checkpoint_dir}/config.json', 'r') as f:
    config = json.load(f)

# 4. Load metrics
with open(f'{checkpoint_dir}/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Loaded {config['model_name']} with RMSE {metrics['rmse']:.2f}")
```

## Checkpoint Contents

### model.keras
Trained Keras model in SavedModel format. Load with `keras.models.load_model()`.

### scaler.pkl
Fitted `StandardScaler` from scikit-learn. Required to normalize input features the same way as training data.

**Important**: Always use the saved scaler for inference, never fit a new one on test data!

### config.json
Complete training configuration:

```json
{
  "units": 64,
  "dense_units": 32,
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 30,
  "max_sequence_length": 1000,
  "validation_split": 0.2,
  "patience_early_stop": 10,
  "patience_lr_reduce": 5,
  "use_early_stop": true
}
```

### metrics.json
Test set evaluation metrics:

```json
{
  "rmse": 6.80,
  "mae": 5.49,
  "r2": 0.9006,
  "rmse_normalized": 0.1046,
  "mae_normalized": 0.0844,
  "accuracy_10": 82.70,
  "accuracy_15": 92.96,
  "accuracy_20": 99.12,
  "phm_score_normalized": 0.1234
}
```

### rul_scaler.json
RUL normalization parameters used for computing normalized metrics:

```json
{
  "y_min": 0.0,
  "y_max": 65.0
}
```

### metadata.json
Complete model metadata including architecture info, performance, training details, and usage examples.

### README.md
Quick reference guide specific to this checkpoint, with performance metrics and code snippets.

## Integration with W&B

Checkpoint paths are automatically logged to Weights & Biases:

```python
wandb.log({"checkpoint/saved_path": "models/mstcn_20260305_143022"})
wandb.summary.update({"checkpoint/directory": "models/mstcn_20260305_143022"})
```

View checkpoint location in W&B run summary or logs.

## Best Practices

### Checkpoint Naming

Checkpoints use timestamped directories to prevent overwrites:
```
models/<model_name>_<YYYYMMDD_HHMMSS>/
```

This ensures every training run is preserved.

### Disk Space Management

Checkpoints are ~5-10 MB each for small models, ~50-100 MB for large models.

To clean up old checkpoints:

```bash
# List checkpoints sorted by date
ls -lt models/

# Remove old checkpoints (keep last 5)
ls -t models/mstcn_* | tail -n +6 | xargs rm -rf
```

### Production Deployment

For production, copy best checkpoint to canonical location:

```bash
# After finding best checkpoint
cp -r models/mstcn_20260305_143022/ models/production/mstcn_model/

# Rename model file for predict.py compatibility
mv models/production/mstcn_model/model.keras \
   models/production/mstcn_model.keras
```

### Versioning

For version control, save metadata.json and README.md to git:

```bash
# Save checkpoint metadata to git (not the large model file)
git add models/mstcn_20260305_143022/metadata.json
git add models/mstcn_20260305_143022/README.md
git add models/mstcn_20260305_143022/config.json
git add models/mstcn_20260305_143022/metrics.json
git commit -m "Add MSTCN checkpoint metadata (RMSE 6.80)"
```

Add to `.gitignore`:
```
# Ignore large model files, keep metadata
models/**/*.keras
models/**/*.pkl
```

## Troubleshooting

### "FileNotFoundError: model.keras not found"

**Cause**: Checkpoint directory doesn't exist or wrong path.

**Solution**:
1. Check checkpoint was saved (look for "Model checkpoint saved successfully")
2. Verify path: `ls models/`
3. Use full path, not relative

### "Model loading failed with incompatible architecture"

**Cause**: Model was saved with different TensorFlow/Keras version.

**Solution**:
1. Check versions: `pip list | grep -E "tensorflow|keras"`
2. Use same environment as training
3. If needed, retrain with current version

### "Predictions don't match training results"

**Cause**: Not using saved scaler, or using different preprocessing.

**Solution**:
1. Always use scaler.pkl from checkpoint
2. Never fit new scaler on test data
3. Check max_sequence_length matches config.json

## Migration from Old Code

### Before (No Auto-Save)

```python
# Old way - model discarded after training
model, history, metrics = train_model(
    dev_X=dev_X, dev_y=dev_y, model_name='mstcn'
)
# Model only exists in memory, lost after script exits
```

### After (Auto-Save Enabled)

```python
# New way - model automatically saved
model, history, metrics = train_model(
    dev_X=dev_X, dev_y=dev_y, model_name='mstcn'
)
# Checkpoint saved to: models/mstcn_<timestamp>/
# Can reload anytime with predict.py
```

### Gradual Migration

If you need old behavior temporarily:

```bash
# Disable auto-save for compatibility
python train_model.py --model mstcn --no-save
```

## Related Features

- **Ensemble Preparation**: See [ENSEMBLE_GUIDE.md](../ENSEMBLE_GUIDE.md)
  - `scripts/prepare_ensemble.py` uses checkpointing to save top 3 models

- **Production API**: See [examples/README.md](../examples/README.md)
  - API server loads checkpoints for inference

- **Cross-Dataset Validation**: See [scripts/cross_dataset_validation.py](../scripts/cross_dataset_validation.py)
  - Depends on checkpointing for model reuse

## Implementation Details

### Function: `save_model_checkpoint()`

Location: `train_model.py` (lines 140-294)

**Inputs**:
- model: Trained Keras model
- scaler: Fitted StandardScaler
- config: Training configuration dict
- test_metrics: Evaluation metrics dict
- model_name: Architecture name
- run_name: Training run name
- y_min, y_max: RUL normalization bounds

**Returns**:
- str: Path to checkpoint directory

**Called from**:
- `train_model()` - After test evaluation, before wandb.finish()
- Only saves if `save_checkpoint=True` and `test_metrics` exists

### Command-Line Flag

```python
parser.add_argument(
    "--no-save",
    action="store_true",
    help="Disable automatic model checkpoint saving",
)
```

Default: Enabled (saves checkpoints)
To disable: Add `--no-save` flag

---

**Last Updated**: March 5, 2026
**Feature Status**: ✅ Production-ready
**Issue**: Closes #21 (Model checkpointing and inference pipeline)
