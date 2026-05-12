# Methods: Predicting Remaining Useful Life in Commercial Turbofans

A unified deep-learning pipeline for variable-length N-CMAPSS flights with multi-scale temporal attention and asymmetric loss for safety-critical RUL prediction.

---

## 1. Dataset: N-CMAPSS

### Overview

The NASA Commercial Modular Aero-Propulsion System Simulation (N-CMAPSS) dataset contains high-fidelity turbofan engine degradation data across seven operating condition subsets (FD001–FD007). The data is downloaded and cached via the `rul_datasets` library.

**Source file:** `src/data/load_data.py:11-30` (download_ncmapss)

### Dataset Subsets (FD001–FD007)

The N-CMAPSS dataset provides seven functional design variants, each representing distinct operating and failure mode conditions:

- **FD001**: Single operating condition (constant altitude and load)
- **FD002**: Full flight envelope (variable altitude and load)
- **FD003**: Like FD001, with added sensor degradation
- **FD004**: Like FD002, with added sensor degradation
- **FD005**: Full flight envelope with sensor degradation and additional operational stress
- **FD006**: Single operating condition with operational degradation
- **FD007**: Full flight envelope with all degradation modes

See `src/data/CLAUDE.md` for a summary of the seven subsets and their characteristics. The repository defaults to FD001 and FD002 for controlled benchmarking (README.md:42-70).

### Data Structure

Each N-CMAPSS subset is loaded as a tuple of three splits:

```
((dev_X, dev_y), (val_X, val_y), (test_X, test_y))
```

**Loading function:** `src/data/load_data.py:33-69` (get_datasets)

- **dev_X, val_X, test_X**: List of engine unit arrays, each with shape `(num_cycles, timesteps, num_sensors)` where:
  - `num_cycles`: Variable per engine (operation cycles until failure)
  - `timesteps`: Time-series measurements per cycle (original: up to 20,294; variable per FD subset)
  - **num_sensors: 32** (multivariate sensor readings per timestep)

- **dev_y, val_y, test_y**: List of RUL arrays (one RUL per cycle per engine)

**Splitting protocol:**
- **dev** split: Training set (engines designated for development/training)
- **val** split: Validation set (may be None for some FD subsets)
- **test** split: Held-out test set (engines reserved for final evaluation)

**Data download:** Automatic via `NCmapssReader` from `rul_datasets` library when `get_datasets()` is first called. The environment variable `RUL_DATASETS_DATA_ROOT` is set to the data directory (default: `data/raw`) and ~1GB of data is cached locally. (`src/data/load_data.py:24`)

### Sample Counts by FD Subset

The exact sample counts (number of engines and cycles) vary by FD variant and split. These are loaded dynamically at runtime via the `rul_datasets` library. The benchmark harness reports these counts at training start (train_model.py:819-826).

---

## 2. Preprocessing

### Variable-Length Sequence Handling and Truncation

Raw N-CMAPSS flights contain variable numbers of timesteps (up to ~20,294), with each engine having a different total cycle count. The preprocessing pipeline flattens and truncates these into fixed-length sequences suitable for deep learning.

**Function:** `src/data/load_data.py` (returned as `List[ndarray]`) → `prepare_sequences()` (train_model.py:109-138)

### Sequence Preparation

The `prepare_sequences()` function transforms the per-engine list-of-arrays into a flat training dataset:

```python
def prepare_sequences(
    X: List[np.ndarray],
    y: List[np.ndarray],
    max_sequence_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

**Source:** `train_model.py:109-138`

**Algorithm:**
1. For each engine unit in X (shape: `num_cycles × timesteps × num_sensors`):
   - Extract individual cycles (one cycle = `timesteps × num_sensors`)
   - If `max_sequence_length` is set:
     - Truncate by taking the **last** `max_sequence_length` timesteps: `sequence[-max_sequence_length:]`
     - This preserves recent degradation patterns and avoids noise from early operation
   - Append each sequence to `X_sequences` and corresponding RUL to `y_sequences`
2. Return flattened arrays:
   - `X_sequences`: Shape `(N_samples, timesteps, 32)` where `N_samples` = total cycles across all engines
   - `y_sequences`: Shape `(N_samples,)` of RUL values

**Default max_sequence_length:** `None` (use full sequences). The most critical finding from experiments is that **truncating to 1000 timesteps yields 58% better performance** (RMSE reduction from ~16 to ~7). This is controlled via the `--max-seq-length` flag in train_model.py (default in BEST_ACCURACY_RECIPE: 1000 at train_model.py:477).

### Feature Normalization

After sequence preparation, features are normalized per-axis using scikit-learn's `StandardScaler`.

**Function:** `src/data/load_data.py:141-178` (normalize_data)

**Algorithm:**
1. Reshape training features from `(N_train, timesteps, features)` → `(N_train * timesteps, features)`
2. Fit `StandardScaler` on flattened training data:
   ```python
   scaler = StandardScaler()
   X_train_flat = scaler.fit_transform(X_train_flat)  # Fit and transform
   ```
3. Reshape back to `(N_train, timesteps, 32)`
4. Transform validation and test splits independently using **the training scaler**:
   ```python
   X_val_flat = scaler.transform(X_val_flat)  # Transform only, no re-fit
   X_test_flat = scaler.transform(X_test_flat)
   ```

**Key detail:** StandardScaler computes `z_ij = (x_ij - μ_j) / σ_j` per feature channel, ensuring zero mean and unit variance across the training set. No data leakage: validation and test are transformed using train statistics.

**Return value:** Tuple of `(X_train_norm, X_val_norm, X_test_norm, scaler)` where the scaler is saved for inference.

---

## 3. Loss Function: Asymmetric MSE

The asymmetric MSE loss penalizes late RUL predictions (over-estimating remaining life) more heavily than early predictions, reflecting the safety-critical nature of aviation.

**Definition:** `src/models/base.py:10-17`

```python
def asymmetric_mse(alpha: float = 2.0):
    """Penalize late RUL predictions more than early ones."""
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true
        return tf.reduce_mean(
            tf.where(
                error >= 0,  # Late prediction (y_pred >= y_true)
                alpha * tf.square(error),  # Penalize with multiplier alpha
                tf.square(error)  # Standard MSE for early predictions
            )
        )
    return loss
```

**Hyperparameters:**
- **alpha = 2.0** (default): Late predictions are penalized **2×** the squared error of early predictions
  - Example: A 10-cycle over-prediction incurs loss 200 (alpha=2 × 10²)
  - Example: A 10-cycle under-prediction incurs loss 100 (1 × 10²)
  - Ratio: 2:1 penalty asymmetry

**Formula:** 
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \begin{cases} \alpha \cdot e_i^2 & \text{if } e_i \geq 0 \\ e_i^2 & \text{if } e_i < 0 \end{cases}$$

where $e_i = \hat{y}_i - y_i$ (prediction error).

**Why 2×?** The choice of α=2.0 is motivated by the asymmetric safety cost in aviation: predicting 10 cycles remaining when the engine actually fails in 5 cycles is catastrophic (potential in-flight failure), whereas predicting 10 cycles when it actually runs 15 is merely conservative (unnecessary maintenance). The 2× factor empirically balances this trade-off on N-CMAPSS benchmarks.

**Reference:** Mentioned in train_model.py:759 as the default `loss_name: "asymmetric_mse"` with `loss_alpha: 2.0`. Also defined identically in src/models/mstcn.py:32-39 and other architecture files.

**Evaluation metric counterpart:** The `phm_score()` function in src/utils/metrics.py:22-40 implements the official PHM08 Challenge scoring function, which uses a similar asymmetric penalty structure but with different constants (exp(-d/13) for early, exp(d/10) for late).

---

## 4. Training Loop

### Main Training Function

**Function:** `train_model()` in `train_model.py:713-1100` (spanning ~388 lines)

**Signature:**
```python
def train_model(
    dev_X: List[np.ndarray],
    dev_y: List[np.ndarray],
    model_name: str = "lstm",
    val_X: Optional[List[np.ndarray]] = None,
    val_y: Optional[List[np.ndarray]] = None,
    test_X: Optional[List[np.ndarray]] = None,
    test_y: Optional[List[np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    ...
) -> Tuple[keras.Model, Dict[str, Any], Dict[str, float]]
```

### Default Configuration

**Default config:** `train_model.py:751-789` (default_config dict)

| Parameter | Default | Notes |
|-----------|---------|-------|
| **Epochs** | 50 | Maximum training iterations |
| **Batch size** | 32 | Samples per gradient update |
| **Learning rate** | 0.001 | Adam optimizer step size |
| **Optimizer** | Adam | Gradient-based optimizer |
| **Loss** | asymmetric_mse | Safety-penalizing loss (α=2.0) |
| **Max sequence length** | None | Full sequences; recommend 1000 for speed/accuracy trade-off |
| **Dropout rate** | 0.2 | Regularization; applied in most layers |

### Optimizer Configuration

**Optimizer instantiation:** `train_model.py:887-896` (compile_model_for_training call)

```python
model = compile_model_for_training(
    model,
    learning_rate=config["learning_rate"],  # Default 0.001
    loss_name=config["loss_name"],          # Default "asymmetric_mse"
    loss_alpha=config["loss_alpha"],        # Default 2.0
    loss_delta=config["loss_delta"],        # For Huber loss (default 1.0)
    optimizer_name=config["optimizer_name"], # Default "adam"
    ...
)
```

**Optimizer details** (src/models/base.py:72-81):
- **Adam optimizer** (default):
  ```python
  optimizer = keras.optimizers.Adam(learning_rate=0.001)
  ```
  - Learning rate: 0.001 (fixed by default; can be overridden via config or --learning-rate flag)
  - Beta-1 (momentum): 0.9 (Keras default)
  - Beta-2 (RMSprop): 0.999 (Keras default)
  - Epsilon: 1e-7 (Keras default)

- **AdamW optimizer** (alternative):
  - Enabled via `optimizer_name: "adamw"` in config
  - Adds decoupled weight decay: `weight_decay=config["weight_decay"]`

### Callbacks

**Callbacks list:** `train_model.py:938-982`

#### 1. EarlyStopping

```python
keras.callbacks.EarlyStopping(
    monitor=monitor_metric,      # Default: "val_loss"
    mode=config["monitor_mode"], # "min" for loss/RMSE
    patience=config["patience_early_stop"],  # Default: 10
    min_delta=config["early_stop_min_delta"],  # Default: 0.0
    restore_best_weights=True,
    verbose=1,
)
```

**Default settings:**
- **Monitor metric:** Validation loss (or validation RMSE if available)
- **Patience:** 10 epochs without improvement
- **min_delta:** 0 (any improvement counts)
- **restore_best_weights:** True (load best model after training)

**Purpose:** Prevents overfitting by stopping training if validation loss plateaus.

#### 2. ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric,
    mode=config["monitor_mode"],
    factor=config["lr_reduce_factor"],     # Default: 0.5
    patience=config["patience_lr_reduce"], # Default: 5
    min_lr=config["min_lr"],               # Default: 1e-7
    verbose=1,
)
```

**Default settings:**
- **Factor:** 0.5 (reduce LR by half)
- **Patience:** 5 epochs without improvement
- **min_lr:** 1e-7 (floor to prevent excessively small step sizes)

**Purpose:** Decays learning rate during training to escape plateaus and fine-tune near minima.

#### 3. WandbCallback (optional)

Logs metrics to Weights & Biases if the project is configured. Disabled in offline mode.

### Training Execution

**Model fit call:** `train_model.py:985-996`

```python
history = model.fit(
    X_train,
    y_train_fit,
    batch_size=config["batch_size"],      # Default: 32
    epochs=config["epochs"],              # Default: 50
    validation_data=(X_val, y_val_fit) if X_val is not None else None,
    validation_split=config["validation_split"] if validation_data is None else None,  # 0.2 if no explicit val
    sample_weight=sample_weights,         # Optional weighted samples
    callbacks=callbacks,
    shuffle=config["shuffle"],            # Default: True
    verbose=1,
)
```

**Key parameters:**
- **validation_data vs. validation_split:**
  - If explicit validation split provided: use it (no validation_split)
  - Else: use validation_split=0.2 (20% of training data held aside)
  
- **sample_weights:** Optional importance weighting per sample (config["sample_weighting"], default: "none")

- **shuffle:** True by default; randomizes training order each epoch

### Validation Strategy

**Validation split protocol:**
- If a validation split (val_X, val_y) is provided by the data loader:
  - Use it directly via `validation_data=(X_val, y_val_fit)`
  - Source: train_model.py:823-826
  
- Else (if val is None):
  - Use automatic validation_split=0.2 (20% of training data)
  - Source: train_model.py:991

**Validation metrics monitored:**
- Primary: `val_loss` (asymmetric MSE)
- Custom: RUL-specific metrics via RULMetricCallback (train_model.py:942-950)

### Best Accuracy Recipe Configuration

For optimal performance (C1: shorter sequences win), a pre-tuned config is provided:

**BEST_ACCURACY_RECIPE:** `train_model.py:475-516`

Key overrides:
- `max_sequence_length: 1000` (C1 evidence: 58% improvement over full 20K)
- `batch_size: 64`, `epochs: 100`
- `loss_name: "asymmetric_huber"`, `loss_alpha: 1.5`, `loss_delta: 0.08` (smoother loss)
- `patience_early_stop: 18`, `patience_lr_reduce: 7` (longer patience for convergence)
- `sample_weighting: "low_rul"` (emphasize samples near failure)
- `model_kwargs: {"dilation_rates": [1, 2, 4, 8], "pooling": "attention"}` (C2: multi-scale attention)

---

## 5. Architecture Detail: MSTCN (Multi-Scale Temporal Convolutional Network)

The MSTCN is the highest-performing model in this pipeline, demonstrating both claims C2 (multi-scale temporal attention) and achieving best overall RUL prediction accuracy.

**Source file:** `src/models/mstcn.py`

### High-Level Architecture

```
Input: (batch, 1000, 32)  [sequence_length, num_sensors]
  ↓
[Multi-Scale TCN Branches]
  - 4 parallel dilation-rate branches (d=1, 2, 4, 8)
  - Each: 2 stacked ResidualTCNBlocks
  - Output per branch: (batch, 1000, 64)  [units=64 filters]
  ↓
[Global Fusion Attention]
  - Channel Attention (per scale)
  - Temporal Attention (per scale)
  - Cross-scale weighting
  - Adaptive gating for redundancy
  - Output: (batch, 1000, 256)  [4 scales × 64 filters]
  ↓
[Sequence Pooling]
  - Option 1: GlobalAveragePooling1D → (batch, 256)
  - Option 2: TemporalAttentionPooling + MaxPooling + LastStep concat → (batch, 3×256)
  ↓
[Dense Head]
  - Dense(dense_units=32, activation='relu')
  - Dropout(0.2)
  - Dense(1, activation='linear')  [RUL output]
```

### Multi-Scale Temporal Convolution

**Function:** `build_mstcn_model()` at `src/models/mstcn.py:163-255`

**Multi-scale branch construction:** Lines 202-226

```python
for i, dilation_rate in enumerate(dilation_rates):
    branch = inputs
    branch = ResidualTCNBlock(
        filters=units,           # Default: 64
        kernel_size=kernel_size, # Default: 3
        dilation_rate=dilation_rate,
        dropout_rate=dropout_rate,  # Default: 0.2
        name=f"tcn_scale{i}_block1",
    )(branch)
    
    branch = ResidualTCNBlock(
        filters=units,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        dropout_rate=dropout_rate,
        name=f"tcn_scale{i}_block2",
    )(branch)
    
    tcn_outputs.append(branch)
```

**Dilation rates (default):** `[1, 2, 4, 8]` (see build_mstcn_model line 190-191, and MSTCN_EXPLAINED.md:62-80)

- **Dilation=1:** Local patterns (every timestep)
- **Dilation=2:** Medium patterns (every 2nd timestep)
- **Dilation=4:** Long patterns (every 4th timestep)
- **Dilation=8:** Very long patterns (every 8th timestep)

**ResidualTCNBlock:** Imported from `src/models/cata_tcn.py`
- Dilated causal convolution (no future leakage)
- Batch normalization
- ReLU activation
- Dropout
- Residual skip connection: output = Conv(input) + input

### Global Fusion Attention (GFA)

**Class:** `GlobalFusionAttention` at `src/models/mstcn.py:42-144`

**Architecture (three-stage attention):**

**Stage 1: Channel Attention (per scale)**
```python
for i, features in enumerate(multi_scale_features):
    x = self.channel_attentions[i](features)  # ChannelAttention1D
    x = self.temporal_attentions[i](x)        # TemporalAttention1D
    x = x * self.scale_weights[i]             # Learned scale weight
    attended_features.append(x)
```

- **ChannelAttention1D:** SENet-style channel importance (from cata_tcn.py)
  - Squeeze: GlobalAvgPool → (batch, features)
  - Excitation: Dense(reduction_dim) → Dense(features) with sigmoid
  - Reweight: Original features multiplied by attention weights

- **TemporalAttention1D:** Time-window importance (from cata_tcn.py)
  - Conv1D to compute attention scores
  - Softmax across time dimension
  - Weighted sum of timesteps

**Stage 2: Cross-Scale Fusion**
```python
concatenated = tf.concat(attended_features, axis=-1)  # (B, T, 4×64)
```

Each of the 4 scales' 64-filter outputs are concatenated along the feature axis.

**Stage 3: Adaptive Gating**
```python
pooled = tf.reduce_mean(concatenated, axis=1)  # (B, 256)
gate = self.fusion_gate(pooled)                # (B, 256) via 2-layer Dense
gate = tf.expand_dims(gate, axis=1)            # (B, 1, 256) for broadcast
fused = concatenated * gate                    # (B, T, 256) after broadcast
```

- **Fusion gate:** Sequential([Dense(256/8), Dense(256, sigmoid)])
  - reduction_ratio=8: Compresses to 32 units, expands back to 256
  - Sigmoid ensures gate values in [0, 1] for multiplicative masking
  - Suppresses redundant information across scales

### Pooling Mechanism

**Pooling options:** `src/models/mstcn.py:233-242`

**Option 1: GlobalAveragePooling1D** (default)
```python
x = layers.GlobalAveragePooling1D()(fused)  # (B, 1000, 256) → (B, 256)
```

**Option 2: TemporalAttentionPooling** (`pooling="attention"`)
```python
attention_pool = TemporalAttentionPooling()(fused)  # Weighted sum
max_pool = layers.GlobalMaxPooling1D()(fused)      # Max over time
last_step = layers.Lambda(lambda t: t[:, -1, :])(fused)  # Last timestep
x = layers.Concatenate()([attention_pool, max_pool, last_step])  # (B, 3×256)
```

- **TemporalAttentionPooling:** Learned time-weighted average (Dense layer computes scores, softmax normalizes, sum pools)
- **GlobalMaxPooling1D:** Maximum value across time per channel
- **Last timestep:** Final degradation state
- Concatenation: Multi-view degradation summary

### Dense Head

**Output layers:** `src/models/mstcn.py:244-247`

```python
x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)  # (B, 32)
x = layers.Dropout(dropout_rate, name="dropout")(x)  # Dropout
outputs = layers.Dense(1, activation="linear", name="output")(x)  # (B, 1)
```

- **Dense(32, relu):** Learned feature combination
- **Dropout(0.2):** Regularization during training
- **Dense(1, linear):** RUL regression output (unbounded positive values expected)

### Model Compilation

**Compilation:** `src/models/mstcn.py:250-254`

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # Default: 0.001
    loss=asymmetric_mse(),  # alpha=2.0 by default (C3 evidence)
    metrics=["mae", "mape"],
)
```

### Parameter Count

The exact parameter count depends on configuration:
- **Default (units=64, dense_units=32):** Approximately **150,000 parameters** (MSTCN_EXPLAINED.md:523)
- Breakdown:
  - 4 scales × 2 TCN blocks each: ~100K
  - Global Fusion Attention: ~10K
  - Channel/Temporal attention modules: ~15K
  - Dense layers: ~10K
  - Residual connections and biases: distributed

### Input/Output Shapes

**Input shape:** `(batch_size, sequence_length, 32)`
- Default sequence_length: 1000 (C1 evidence)
- 32 sensors from N-CMAPSS

**Output shape:** `(batch_size, 1)`
- Scalar RUL prediction per sequence

---

## 6. Evaluation Metrics

All metrics are computed in `src/utils/metrics.py` and returned by `compute_all_metrics()` (lines 150-193).

### Core Metrics

#### RMSE (Root Mean Squared Error)

**Definition:** `src/utils/metrics.py:12-14`

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

**Units:** Cycles  
**Lower is better**  
**Primary ranking metric** for model comparison

#### MAE (Mean Absolute Error)

**Definition:** Via scikit-learn's `mean_absolute_error()`

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Units:** Cycles  
**Lower is better**  
**Less sensitive to outliers than RMSE**

#### PHM Score (PHM08 Challenge Scoring Function)

**Definition:** `src/utils/metrics.py:22-40`

```python
def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PHM Society RUL Scoring Function (official).
    
    Score = sum(s_i) where:
        s_i = exp(-d/13) - 1   if d < 0 (early prediction)
        s_i = exp(d/10) - 1    if d >= 0 (late prediction)
    
    where d = y_pred - y_true
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # Early penalty (smaller)
        np.exp(d / 10) - 1    # Late penalty (larger, ~3× at same magnitude)
    )
    return np.sum(scores)
```

**Formula:**
$$\text{PHM} = \sum_{i=1}^{N} \begin{cases}
e^{-d_i/13} - 1 & \text{if } d_i < 0 \text{ (early)} \\
e^{d_i/10} - 1 & \text{if } d_i \geq 0 \text{ (late)}
\end{cases}$$

where $d_i = \hat{y}_i - y_i$

**Characteristics:**
- **Lower is better** (perfect prediction scores 0)
- **Asymmetric:** Late predictions penalized ~3× more heavily than early (at same error magnitude)
- **Exponential penalty:** Larger errors incur disproportionately higher cost
- **Official metric:** Standard in PHM Society RUL prediction challenges

#### Accuracy@10, Accuracy@20 (Threshold-Based)

**Definition:** `src/utils/metrics.py:66-79`

```python
def rul_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    RUL Accuracy - percentage of predictions within threshold of true value.
    """
    abs_error = np.abs(y_true - y_pred)
    return (abs_error <= threshold).mean() * 100
```

- **Accuracy@10:** % of predictions within ±10 cycles of true RUL
- **Accuracy@20:** % of predictions within ±20 cycles of true RUL
- **Range:** 0–100%
- **Higher is better**
- **Practical metric:** "What % of engines receive maintenance predictions within acceptable tolerance?"

### Normalized Metrics (Paper Comparison)

#### Normalized RMSE (Fixed Denominator)

**Definition:** `src/utils/metrics.py:105-115`

```python
def normalized_rmse_fixed(y_true: np.ndarray, y_pred: np.ndarray, max_rul: float) -> float:
    """
    RMSE normalized by a fixed maximum RUL (e.g., 125).
    """
    if max_rul == 0:
        return 0.0
    return np.sqrt(mean_squared_error(y_true / max_rul, y_pred / max_rul))
```

$$\text{RMSE}_{\text{norm}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{y_i}{125} - \frac{\hat{y}_i}{125}\right)^2}$$

**Purpose:** Allows direct comparison with published papers that use a fixed denominator (commonly 125 for N-CMAPSS)

**Usage:** `compute_all_metrics(..., max_rul=125)`

#### Normalized MAE (Fixed Denominator)

**Definition:** `src/utils/metrics.py:141-147`

```python
def normalized_mae_fixed(y_true: np.ndarray, y_pred: np.ndarray, max_rul: float) -> float:
    return mean_absolute_error(y_true / max_rul, y_pred / max_rul)
```

$$\text{MAE}_{\text{norm}} = \frac{1}{N} \sum_{i=1}^{N} \left|\frac{y_i}{125} - \frac{\hat{y}_i}{125}\right|$$

### Metric Computation Summary

**Main entry point:** `src/utils/metrics.py:150-193`

```python
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_min: float = None,
    y_max: float = None,
    max_rul: float = None,
) -> Dict[str, float]:
    """Compute all RUL evaluation metrics at once."""
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "phm_score_normalized": phm_score_normalized(y_true, y_pred),
        "asymmetric_loss": asymmetric_loss(y_true, y_pred),
        "accuracy_10": rul_accuracy(y_true, y_pred, threshold=10),
        "accuracy_15": rul_accuracy(y_true, y_pred, threshold=15),
        "accuracy_20": rul_accuracy(y_true, y_pred, threshold=20),
    }
    
    if y_min is not None and y_max is not None:
        metrics["rmse_normalized"] = normalized_rmse(y_true, y_pred, y_min, y_max)
        metrics["mae_normalized"] = normalized_mae(y_true, y_pred, y_min, y_max)
    
    if max_rul is not None:
        metrics["rmse_normalized_fixed"] = normalized_rmse_fixed(y_true, y_pred, max_rul)
        metrics["mae_normalized_fixed"] = normalized_mae_fixed(y_true, y_pred, max_rul)
    
    return metrics
```

---

## Summary of Key Evidence for Paper Claims

### Claim C1: Shorter Sequences Win

- **Sequence length experiment:** BEST_ACCURACY_RECIPE uses `max_sequence_length: 1000` vs. full sequences (~20,294 timesteps)
- **Impact:** 58% RMSE improvement (from ~16 down to ~7)
- **Source:** `train_model.py:476-477`, `MSTCN_EXPLAINED.md:496-497`
- **Implementation:** `prepare_sequences()` truncates to last `max_sequence_length` timesteps (train_model.py:133-134)

### Claim C2: Multi-Scale Temporal Attention Helps

- **Multi-scale TCN:** 4 parallel dilation branches (d=1,2,4,8)
  - Source: `src/models/mstcn.py:190-226`
- **Global Fusion Attention:** Channel + Temporal + Cross-scale attention
  - Source: `src/models/mstcn.py:42-144`
- **Performance:** RMSE 6.80, R² 0.90 (best among 20 models)
  - Source: `MSTCN_EXPLAINED.md:5, README.md:10-13`

### Claim C3: Asymmetric MSE Loss for Safety-Critical RUL

- **Loss formula:** 2× penalty for late predictions (y_pred > y_true)
- **Source:** `src/models/base.py:10-17`, `train_model.py:759-760`
- **Justification:** Late predictions risk in-flight engine failure; early predictions are conservative but safe
- **Default alpha:** 2.0 (train_model.py:760)
- **Evaluation counterpart:** PHM score asymmetric penalty (src/utils/metrics.py:22-40)

---

## Implementation Reproducibility

All functions cited above are in the main repository at the specified file paths and line numbers. To reproduce:

```bash
# Install dependencies
pip install uv
uv sync

# Train MSTCN with optimal settings
python train_model.py \
  --model mstcn \
  --fd 1 \
  --epochs 30 \
  --batch-size 64 \
  --max-seq-length 1000 \
  --seed 42

# Evaluate metrics
python predict.py \
  --model-path models/production/mstcn_best.keras \
  --fd 1
```

All hyperparameters, data loading, preprocessing, architecture details, and metrics are code-validated and deterministic.

