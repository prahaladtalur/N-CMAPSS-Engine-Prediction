# Model Selection Guide for RUL Prediction

This guide helps you choose the right model architecture for your RUL (Remaining Useful Life) prediction task.

## Quick Start

```bash
# List all available models
python train_model.py --list-models

# Get recommendations for your use case
python train_model.py --recommend

# Train a single model
python train_model.py --model lstm

# Compare multiple models
python train_model.py --compare --models lstm gru transformer

# Compare ALL models (takes time!)
python train_model.py --compare-all
```

---

## Available Models (13 Total)

### **RNN-based Models** 🔄

#### 1. **LSTM** (Long Short-Term Memory)
- **Best for:** Standard baseline, interpretable results
- **Pros:** Well-understood, stable training, good for moderate sequences
- **Cons:** Slower than GRU, can struggle with very long sequences
- **Use when:** You want a reliable baseline or need interpretability
- **Typical performance:** ⭐⭐⭐☆☆

```python
python train_model.py --model lstm
```

#### 2. **BiLSTM** (Bidirectional LSTM)
- **Best for:** When future context helps (less applicable for real-time RUL)
- **Pros:** Captures patterns in both directions
- **Cons:** 2x slower than LSTM, not suitable for real-time prediction
- **Use when:** Training offline models where you have full sequences
- **Typical performance:** ⭐⭐⭐⭐☆

```python
python train_model.py --model bilstm
```

#### 3. **GRU** (Gated Recurrent Unit)
- **Best for:** Faster training, limited data
- **Pros:** Faster than LSTM, fewer parameters, often similar performance
- **Cons:** Slightly less expressive than LSTM
- **Use when:** You want faster training or have limited data
- **Typical performance:** ⭐⭐⭐☆☆

```python
python train_model.py --model gru --epochs 50
```

#### 4. **BiGRU** (Bidirectional GRU)
- **Best for:** Faster bidirectional model
- **Pros:** Faster than BiLSTM, good performance
- **Cons:** Not for real-time prediction
- **Use when:** You want BiLSTM performance but faster
- **Typical performance:** ⭐⭐⭐⭐☆

#### 5. **Attention LSTM**
- **Best for:** Identifying critical time steps in degradation
- **Pros:** Focuses on important time steps, interpretable attention weights
- **Cons:** More parameters, slightly slower
- **Use when:** You want to understand which time steps matter most
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model attention_lstm --units 128
```

#### 6. **ResNet-LSTM**
- **Best for:** Deep networks, complex patterns
- **Pros:** Better gradient flow, can use more layers
- **Cons:** More complex, longer training time
- **Use when:** You need a very deep network for complex patterns
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model resnet_lstm --epochs 100
```

---

### **Convolutional Models** 🌊

#### 7. **TCN** (Temporal Convolutional Network)
- **Best for:** Long sequences, parallelizable training
- **Pros:** Very fast training, excellent receptive field, parallelizable
- **Cons:** More memory usage
- **Use when:** You have long sequences or want fast training
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model tcn --units 64
```

#### 8. **WaveNet**
- **Best for:** Long-range dependencies, complex temporal patterns
- **Pros:** Gated activations, excellent for time series, large receptive field
- **Cons:** More parameters, can be slower
- **Use when:** You have long sequences with complex dependencies
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model wavenet --epochs 100
```

---

### **Hybrid Models** 🔀

#### 9. **CNN-LSTM**
- **Best for:** Feature extraction + temporal modeling
- **Pros:** CNN learns spatial patterns, LSTM models time
- **Cons:** More complex, slower training
- **Use when:** Raw sensor data has spatial structure
- **Typical performance:** ⭐⭐⭐⭐☆

```python
python train_model.py --model cnn_lstm
```

#### 10. **CNN-GRU**
- **Best for:** Faster hybrid model
- **Pros:** Similar to CNN-LSTM but faster
- **Cons:** Slightly less expressive than CNN-LSTM
- **Use when:** You want CNN-LSTM speed improvements
- **Typical performance:** ⭐⭐⭐⭐☆

```python
python train_model.py --model cnn_gru --batch-size 64
```

#### 11. **Inception-LSTM**
- **Best for:** Multi-scale feature extraction
- **Pros:** Captures patterns at multiple time scales simultaneously
- **Cons:** More parameters, complex architecture
- **Use when:** Degradation occurs at multiple time scales
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model inception_lstm --units 128
```

---

### **Attention-based Models** 🎯

#### 12. **Transformer**
- **Best for:** State-of-the-art performance, long sequences
- **Pros:** Self-attention, captures long-range dependencies, parallelizable
- **Cons:** Requires more data, more parameters
- **Use when:** You have sufficient data and want best performance
- **Typical performance:** ⭐⭐⭐⭐⭐

```python
python train_model.py --model transformer --units 128 --epochs 100
```

---

### **Baseline** 📊

#### 13. **MLP** (Multi-Layer Perceptron)
- **Best for:** Baseline comparison
- **Pros:** Very fast, simple, no temporal modeling
- **Cons:** Ignores temporal structure, poor performance
- **Use when:** You need a quick baseline or sanity check
- **Typical performance:** ⭐⭐☆☆☆

```python
python train_model.py --model mlp --epochs 30
```

---

## Model Selection Decision Tree

```
START
  |
  ├─ Need quick baseline?
  │   └─ YES → GRU or MLP
  │
  ├─ Have limited data (<10k samples)?
  │   └─ YES → GRU or LSTM
  │
  ├─ Have very long sequences (>500 timesteps)?
  │   └─ YES → TCN, WaveNet, or Transformer
  │
  ├─ Want best accuracy (have time + data)?
  │   └─ YES → Transformer, Attention-LSTM, WaveNet, or ResNet-LSTM
  │
  ├─ Need interpretability?
  │   └─ YES → LSTM or Attention-LSTM
  │
  ├─ Want fastest training?
  │   └─ YES → GRU, CNN-GRU, or TCN
  │
  └─ Have multi-scale patterns?
      └─ YES → Inception-LSTM, Transformer, or WaveNet
```

---

## Recommendations by Use Case

### 🚀 **Quick Experiments**
**Recommended:** GRU, MLP

```bash
python train_model.py --model gru --epochs 30
```

### 🏆 **Best Accuracy** (Production)
**Recommended:** Transformer, Attention-LSTM, WaveNet, ResNet-LSTM

```bash
# Compare top models
python train_model.py --compare --models transformer attention_lstm wavenet resnet_lstm --epochs 100
```

### ⚡ **Fastest Training**
**Recommended:** GRU, CNN-GRU, TCN

```bash
python train_model.py --model tcn --batch-size 64
```

### 🔍 **Most Interpretable**
**Recommended:** LSTM, Attention-LSTM

```bash
python train_model.py --model attention_lstm
# Attention weights show which timesteps are important
```

### 📏 **Long Sequences** (>500 timesteps)
**Recommended:** TCN, WaveNet, Transformer

```bash
python train_model.py --model wavenet --max-seq-length 1000
```

### 📊 **Limited Data** (<10k samples)
**Recommended:** GRU, LSTM

```bash
python train_model.py --model gru --dropout 0.3
```

### 🎨 **Complex Multi-Scale Patterns**
**Recommended:** Transformer, Inception-LSTM, WaveNet

```bash
python train_model.py --model inception_lstm --units 128
```

---

## Performance Comparison

Based on typical RUL prediction tasks:

| Model | Speed | Accuracy | Memory | Data Need | Interpretability |
|-------|-------|----------|--------|-----------|------------------|
| **LSTM** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **BiLSTM** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **GRU** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **BiGRU** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Attention LSTM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ResNet-LSTM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **TCN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **WaveNet** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **CNN-LSTM** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **CNN-GRU** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Inception-LSTM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Transformer** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **MLP** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Training Tips

### For Best Results:

1. **Start with GRU** - Quick baseline to validate your pipeline
2. **Try Transformer or Attention-LSTM** - Usually best accuracy
3. **Experiment with TCN/WaveNet** - Often outperform RNNs
4. **Use MLP as sanity check** - Should perform worse (validates temporal importance)

### Hyperparameter Tuning:

```bash
# Experiment with units
python train_model.py --model transformer --units 128

# Adjust learning rate
python train_model.py --model wavenet --lr 0.0005

# Increase epochs for complex models
python train_model.py --model resnet_lstm --epochs 150

# Try different batch sizes
python train_model.py --model tcn --batch-size 64
```

### Model Comparison:

```bash
# Compare top 3 models
python train_model.py --compare --models transformer attention_lstm wavenet --epochs 100

# Compare RNN variants
python train_model.py --compare --models lstm bilstm gru bigru

# Compare all (takes time!)
python train_model.py --compare-all --epochs 30
```

---

## Advanced Usage

### Custom Configuration:

```python
from src.models.train import train_model
from src.data.load_data import get_datasets

# Load data
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

# Custom config
config = {
    "units": 128,
    "dense_units": 64,
    "dropout_rate": 0.3,
    "learning_rate": 0.0005,
    "batch_size": 64,
    "epochs": 150,
}

# Train model
model, history, metrics = train_model(
    dev_X=dev_X,
    dev_y=dev_y,
    model_name="transformer",
    test_X=test_X,
    test_y=test_y,
    config=config,
)
```

### Model-Specific Parameters:

```python
from src.models.architectures import get_model

# Transformer with more heads
model = get_model("transformer", input_shape=(50, 14), num_heads=8, num_layers=4)

# TCN with more layers
model = get_model("tcn", input_shape=(50, 14), num_layers=6, kernel_size=5)

# WaveNet with custom depth
model = get_model("wavenet", input_shape=(50, 14), num_layers=10)
```

---

## Troubleshooting

### Model not converging?
- **Try:** Lower learning rate (`--lr 0.0001`)
- **Try:** Add more dropout (`--dropout 0.3`)
- **Try:** Reduce model complexity (`--units 32`)

### Training too slow?
- **Try:** GRU, CNN-GRU, or TCN
- **Try:** Increase batch size (`--batch-size 64`)
- **Try:** Reduce units (`--units 32`)

### Overfitting?
- **Try:** Increase dropout (`--dropout 0.3`)
- **Try:** Reduce model size (`--units 32 --dense-units 16`)
- **Try:** More regularization
- **Try:** Simpler model (GRU instead of Transformer)

### Underfitting?
- **Try:** More complex model (Transformer, ResNet-LSTM)
- **Try:** Increase capacity (`--units 128 --dense-units 64`)
- **Try:** Train longer (`--epochs 200`)
- **Try:** Lower dropout (`--dropout 0.1`)

---

## References

- **LSTM/GRU:** Standard RNN architectures for sequence modeling
- **Attention:** "Attention is All You Need" (Vaswani et al., 2017)
- **TCN:** "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)
- **WaveNet:** "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
- **ResNet:** "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **Inception:** "Going Deeper with Convolutions" (Szegedy et al., 2015)

---

**For more information:**
- See `python train_model.py --help`
- Check model implementations in `src/models/architectures.py`
- Review training pipeline in `src/models/train.py`
