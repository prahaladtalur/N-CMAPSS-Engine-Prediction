# Loss Functions for RUL Prediction

Standard MSE loss treats all errors equally, but for RUL prediction, **not all errors are equal**:
1. ❌ Errors in critical zones (low RUL) are more costly
2. ❌ Under-prediction (predicting too late) can cause catastrophic failures
3. ❌ Standard metrics don't reflect real-world costs

This guide explains the custom loss functions designed specifically for RUL prediction.

---

## Why Custom Loss Functions Matter

### The Problem with Standard MSE

```python
# Standard MSE treats these equally:
Error at RUL=200: Predicted 210, Actual 200 → Error = 10 cycles
Error at RUL=20:  Predicted 30,  Actual 20  → Error = 10 cycles

# But the second error is MUCH more dangerous!
# At RUL=20, you're in the critical zone - accuracy matters way more
```

### Real-World Cost Asymmetry

| Prediction Type | Impact | Cost |
|----------------|--------|------|
| **Under-prediction** (pred > actual) | Predict failure too late | 💥 **Catastrophic failure** |
| **Over-prediction** (pred < actual) | Predict failure too early | 💰 Unnecessary maintenance |

**Under-prediction is WAY worse!** This is why we need asymmetric loss functions.

---

## Available Loss Functions

### 1. **Combined RUL Loss** (⭐ RECOMMENDED)

Combines critical zone weighting + asymmetric penalty.

```bash
# Use in CLI (default)
python train_model.py --model transformer --loss combined_rul

# In code
from src.models.losses import combined_rul_loss
loss = combined_rul_loss(
    critical_threshold=30,      # RUL < 30 = critical
    critical_weight=3.0,         # 3x penalty in critical zone
    under_prediction_weight=2.0, # 2x penalty for under-prediction
    alpha=0.7                    # Balance with standard MSE
)
```

**What it does:**
- Errors at RUL < 30 get 3x weight
- Under-predictions get 2x weight
- Combined: Critical + under-predicted errors get 6x weight!

**Best for:** Production systems, overall best performance

---

### 2. **PHM Score Loss**

Official Prognostics and Health Management competition scoring.

```bash
python train_model.py --model lstm --loss phm_score
```

```python
from src.models.losses import phm_score_loss

# Asymmetric exponential penalty:
# Under-prediction: e^(error/10) - 1
# Over-prediction:  e^(-error/13) - 1
```

**What it does:**
- Exponentially penalizes late predictions
- Less severe for early predictions
- Based on PHM08 competition standard

**Best for:** Research, competitions, comparing with published results

---

### 3. **Weighted MSE Loss**

Focus on critical zone accuracy.

```bash
python train_model.py --model gru --loss weighted_mse
```

```python
from src.models.losses import weighted_mse_loss

loss = weighted_mse_loss(
    critical_threshold=30,  # Critical zone threshold
    critical_weight=3.0     # Weight multiplier for critical zone
)
```

**What it does:**
- Standard MSE with higher weight for low RUL values
- Errors at RUL < 30 count 3x more

**Best for:** When you specifically need accuracy in critical zones

---

### 4. **Asymmetric MSE Loss**

Penalize under-prediction more than over-prediction.

```bash
python train_model.py --model tcn --loss asymmetric_mse
```

```python
from src.models.losses import asymmetric_mse_loss

loss = asymmetric_mse_loss(
    under_prediction_weight=2.0  # Under-prediction penalty multiplier
)
```

**What it does:**
- Under-predictions (pred > true) get 2x penalty
- Over-predictions (pred < true) get 1x penalty
- Encourages conservative (safe) predictions

**Best for:** Maximum safety, avoiding late failure predictions

---

### 5. **Quantile Loss**

For uncertainty-aware conservative estimates.

```bash
python train_model.py --model wavenet --loss quantile_90
```

```python
from src.models.losses import quantile_loss

loss = quantile_loss(quantile=0.9)  # 90th percentile
```

**What it does:**
- Optimizes for specific quantile (e.g., 90th percentile)
- Produces conservative estimates

**Best for:** When you want consistently safe estimates

---

### 6. **Standard MSE** (Baseline)

Original unweighted loss.

```bash
python train_model.py --model lstm --loss mse
```

**What it does:**
- Treats all errors equally
- No special consideration for critical zones or asymmetry

**Best for:** Baseline comparison, sanity checks

---

## Quick Comparison

| Loss Function | Critical Zone Focus | Asymmetric Penalty | Use Case |
|--------------|-------------------|-------------------|----------|
| **combined_rul** ⭐ | ✅ Yes (3x) | ✅ Yes (2x) | Production, best overall |
| **phm_score** | ❌ No | ✅ Yes (exponential) | Competitions, research |
| **weighted_mse** | ✅ Yes (3x) | ❌ No | Critical zone focus |
| **asymmetric_mse** | ❌ No | ✅ Yes (2x) | Maximum safety |
| **quantile_90** | ❌ No | ✅ Yes (implicit) | Conservative estimates |
| **mse** | ❌ No | ❌ No | Baseline |

---

## Usage Examples

### CLI Usage

```bash
# Recommended: Combined RUL loss (default)
python train_model.py --model transformer --loss combined_rul

# PHM competition scoring
python train_model.py --model lstm --loss phm_score --epochs 100

# Focus on critical zone
python train_model.py --model gru --loss weighted_mse

# Maximum safety (conservative)
python train_model.py --model wavenet --loss asymmetric_mse
```

### Programmatic Usage

```python
from src.models.train import train_model
from src.models.losses import combined_rul_loss, get_loss_function
from src.data.load_data import get_datasets

# Load data
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

# Option 1: Use predefined loss
config = {
    "epochs": 100,
    "units": 128,
    "loss": get_loss_function("combined_rul"),
}

# Option 2: Custom loss parameters
config = {
    "epochs": 100,
    "units": 128,
    "loss": combined_rul_loss(
        critical_threshold=20,       # More aggressive critical zone
        critical_weight=5.0,          # 5x weight for critical
        under_prediction_weight=3.0,  # 3x weight for under-prediction
        alpha=0.8                     # Heavier weighting
    ),
}

# Train
model, history, metrics = train_model(
    dev_X=dev_X,
    dev_y=dev_y,
    model_name="transformer",
    test_X=test_X,
    test_y=test_y,
    config=config,
)
```

### Custom Loss Function

```python
from src.models.losses import weighted_mse_loss
from src.models.architectures import get_model

# Create custom loss
my_loss = weighted_mse_loss(
    critical_threshold=15,   # Very strict critical zone
    critical_weight=10.0      # Heavy penalty
)

# Build model with custom loss
model = get_model(
    "lstm",
    input_shape=(50, 14),
    loss=my_loss,
    units=64
)
```

---

## Recommendations by Use Case

### 🏆 Production System (Best Overall)
```bash
python train_model.py --model transformer --loss combined_rul --epochs 150
```
- **Loss:** `combined_rul`
- **Why:** Balances critical zone focus + safety + overall accuracy

### 🔬 Research / Competitions
```bash
python train_model.py --model attention_lstm --loss phm_score --epochs 100
```
- **Loss:** `phm_score`
- **Why:** Standard metric for comparing with published results

### 🚨 Maximum Safety (Avoid Failures)
```bash
python train_model.py --model wavenet --loss asymmetric_mse --epochs 100
```
- **Loss:** `asymmetric_mse` or `quantile_90`
- **Why:** Heavily penalizes late predictions

### 🎯 Critical Zone Focus
```bash
python train_model.py --model tcn --loss weighted_mse --epochs 80
```
- **Loss:** `weighted_mse`
- **Why:** Emphasizes accuracy when RUL is low

### 📊 Baseline / Comparison
```bash
python train_model.py --model gru --loss mse --epochs 50
```
- **Loss:** `mse`
- **Why:** Standard unweighted loss for comparison

---

## Impact Example

Here's what different loss functions optimize for:

```
Engine at RUL=25 (critical zone, actual=25):

Standard MSE:
  Pred=35 (late):  Error = 10² = 100
  Pred=15 (early): Error = 10² = 100
  → Treats both equally ❌

Combined RUL Loss (critical_weight=3, under_weight=2):
  Pred=35 (late):  Error = 10² × 3 × 2 = 600  💥
  Pred=15 (early): Error = 10² × 3 × 1 = 300  ✓
  → Penalizes late prediction 2x more ✅
  → Both get 3x weight for being in critical zone ✅
```

---

## Troubleshooting

### Loss not decreasing?
- **Try:** Start with `mse`, then switch to custom loss
- **Try:** Reduce custom weights (e.g., `critical_weight=2.0` instead of `5.0`)
- **Try:** Adjust `alpha` in combined_rul (lower α = more standard MSE)

### Model too conservative (always over-predicts)?
- **Try:** Reduce `under_prediction_weight`
- **Try:** Use `weighted_mse` instead of `asymmetric_mse`

### Model not conservative enough (under-predicts)?
- **Try:** Increase `under_prediction_weight`
- **Try:** Use `asymmetric_mse` or `quantile_90`

### Poor performance in critical zone?
- **Try:** Increase `critical_weight`
- **Try:** Lower `critical_threshold` (e.g., from 30 to 20)

---

## Technical Details

### Combined RUL Loss Formula

```python
error = predicted - actual
squared_error = error²

# Critical zone weight
critical_weight = 3.0 if actual < threshold else 1.0

# Asymmetric weight
asymmetric_weight = 2.0 if error > 0 else 1.0

# Combined
total_weight = critical_weight × asymmetric_weight
weighted_error = squared_error × total_weight

# Balanced with standard MSE
loss = α × mean(weighted_error) + (1-α) × mean(squared_error)
```

### PHM Score Formula

```python
error = predicted - actual

if error > 0:  # Under-prediction (late)
    penalty = exp(error/10) - 1
else:  # Over-prediction (early)
    penalty = exp(-error/13) - 1

loss = mean(penalty)
```

---

## References

- **PHM Score:** Based on PHM08 Prognostics Competition
- **Asymmetric Loss:** Common in cost-sensitive learning
- **Weighted Loss:** Standard technique for imbalanced importance

---

## Summary

✅ **Use `combined_rul` for production** - best overall performance
✅ **Use `phm_score` for research** - standard competition metric
✅ **Use `asymmetric_mse` for safety** - avoid late predictions
✅ **Use `weighted_mse` for critical focus** - accuracy at low RUL
✅ **Use `mse` for baseline** - comparison and sanity checks

**Default:** The CLI now uses `combined_rul` by default - it's the best choice for most RUL prediction tasks!

```bash
# Just run this - it uses the recommended loss automatically
python train_model.py --model transformer
```
