# src/models/ — Model Architectures

## Purpose

All neural network architectures for RUL prediction. Uses a decorator-based ModelRegistry for easy model switching. See [../../EXPERIMENTS.md](../../EXPERIMENTS.md) for benchmark results.

## Key Files

- **architectures.py** — `ModelRegistry` class, `BaseModel` ABC, 12 built-in model classes, `asymmetric_mse` loss, custom layers (`AttentionLayer`, `TCNBlock`, `WaveNetBlock`)
- **cnn_lstm_attention.py** — 2024 SOTA CNN-LSTM-Attention architecture (`SelfAttentionLayer`, `CNNFeatureExtractor`, `build_cnn_lstm_attention_model`)
- **mdfa.py** — MDFA module (`ChannelAttention`, `SpatialAttention`, `MDFAModule`)
- **__init__.py** — Lazy imports from `train_model.py` to avoid circular deps; re-exports `ModelRegistry`, `get_model`, `list_available_models`

## ModelRegistry API

| Method | Purpose |
|--------|---------|
| `ModelRegistry.register("name")` | Class decorator to register a model |
| `ModelRegistry.build(name, input_shape, **kwargs)` | Build a compiled model by name |
| `ModelRegistry.get(name)` | Get model class by name |
| `ModelRegistry.list_models()` | List all registered names |
| `get_model(name, input_shape, **kwargs)` | Convenience wrapper for `build()` |
| `get_model_info()` (line 939) | Dict of name -> description |
| `get_model_recommendations()` (line 965) | Dict of use_case -> [model_names] |

## BaseModel Contract (line 84)

Every model class must:
1. Inherit from `BaseModel`
2. Implement `static build(input_shape, units=64, dense_units=32, dropout_rate=0.2, learning_rate=0.001) -> keras.Model`
3. Call `BaseModel.compile_model(model, learning_rate)` which sets:
   - Optimizer: Adam
   - Loss: `asymmetric_mse(alpha=2.0)`
   - Metrics: `["mae", "mape"]`

## Adding a New Model

1. Add class to `architectures.py` (or new file, then import it in `architectures.py`)
2. Decorate with `@ModelRegistry.register("my_model")`
3. Implement `build()` following the `BaseModel` signature
4. Call `BaseModel.compile_model()` for compilation
5. Update `get_model_info()` dict (line 939)
6. Update `get_model_recommendations()` dict (line 965) if applicable
7. Test: `python train_model.py --model my_model --epochs 5`
8. Run: `make check`

## Registered Architectures (15 total)

| Category | Models |
|----------|--------|
| RNN | `lstm`, `bilstm`, `gru`, `bigru`, `attention_lstm`, `resnet_lstm` |
| Convolutional | `tcn`, `wavenet` |
| Hybrid | `cnn_lstm`, `cnn_gru`, `inception_lstm` |
| Attention | `transformer`, `mdfa`, `cnn_lstm_attention` |
| Baseline | `mlp` |

## Best Performing (see [../../EXPERIMENTS.md](../../EXPERIMENTS.md))

1. **CNN-GRU** + asymmetric loss: RMSE 6.44, R² 0.91
2. **WaveNet** + asymmetric loss: RMSE 6.73
3. **Transformer** + asymmetric loss: RMSE 6.75

CNN-LSTM does **not** converge. CNN-GRU is the stable hybrid variant.

## asymmetric_mse Loss (line 64)

Penalizes late predictions (`y_pred > y_true`) by `alpha=2.0x`. Over-predicting remaining life is dangerous in aviation — the engine might fail before the predicted end of life. This is the training loss; `phm_score` in [../utils/CLAUDE.md](../utils/CLAUDE.md) is the corresponding evaluation metric.

## Custom Layers

| Layer | File | Purpose |
|-------|------|---------|
| `AttentionLayer` | architectures.py | Additive attention for sequences |
| `TCNBlock` | architectures.py | Dilated causal conv with residual connections |
| `WaveNetBlock` | architectures.py | Gated dilated causal conv with residual |
| `SelfAttentionLayer` | cnn_lstm_attention.py | Scaled dot-product Q/K/V attention |
| `CNNFeatureExtractor` | cnn_lstm_attention.py | Conv1D stack with BatchNorm + pooling |
| `ChannelAttention` | mdfa.py | SENet-style sensor importance weighting |
| `SpatialAttention` | mdfa.py | Time window importance weighting |
| `MDFAModule` | mdfa.py | Multi-scale dilated fusion + dual attention |

## Pitfalls

- `__init__.py` lazy-imports `train_model` functions to avoid circular deps — do not convert to top-level imports
- `cnn_lstm_attention.py` has its own `asymmetric_mse` (duplicated from `architectures.py`) because it uses a standalone build function
- Models expect `input_shape=(timesteps, features)` — no batch dimension
- Some models accept extra kwargs (`num_heads`, `num_layers`, `kernel_size`, `dilation_rates`, `cnn_filters`) — check each `build()` signature
