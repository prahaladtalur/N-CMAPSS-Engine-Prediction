# SOTA Optimization Log

**Goal**: Close the 3-5x performance gap (RMSE_norm 0.098 → 0.021-0.032)  
**Started**: 2026-03-08  
**Target**: Beat published SOTA (RMSE_norm < 0.032)

---

## Current Baseline

### Best Model (from EXPERIMENTS.md)
- **Model**: CNN-GRU + asymmetric loss
- **RMSE (raw)**: 6.44
- **RMSE_norm**: TBD (need to run with normalization)
- **Configuration**:
  - Epochs: 100 (early stopped ~40)
  - Batch size: 32
  - Units: 64
  - Dense units: 32
  - Dropout: 0.2
  - Learning rate: 0.001
  - Sequence length: 1000 timesteps
  - Loss: Asymmetric MSE (α=2.0)

### SOTA Targets (from papers)
- **RMSE_norm**: 0.021-0.032
- **MAE_norm**: 0.026
- **R²**: 0.987

---

## Hypotheses for Performance Gap

1. **Sequence Length**: Currently 1000, papers use 500-2000+
2. **Data Preprocessing**: Z-score vs min-max vs per-engine normalization
3. **RUL Clipping**: Papers clip max RUL at 125-130 cycles
4. **Hyperparameters**: Learning rate, batch size, units not optimized
5. **Training Duration**: May need more epochs with better patience settings
6. **Model Architecture**: May need architecture-specific tuning

---

## Optimization Roadmap

### Phase 1: Quick Wins (2-3 hours)
- [ ] Run benchmark with normalized metrics to establish baseline
- [ ] Try longer sequence lengths (1500, 2000)
- [ ] Experiment with RUL clipping (125, 130)
- [ ] Test different normalization strategies

### Phase 2: Hyperparameter Tuning (1-2 days)
- [ ] Learning rate sweep (0.0001, 0.0005, 0.001, 0.002)
- [ ] Batch size sweep (16, 32, 64, 128)
- [ ] Model capacity sweep (units: 32, 64, 128, 256)
- [ ] Dropout sweep (0.1, 0.2, 0.3, 0.4)

### Phase 3: Training Protocol (1-2 days)
- [ ] Longer training with better early stopping
- [ ] Learning rate schedules (cosine, warmup)
- [ ] Loss function variations (Huber, quantile)
- [ ] Gradient clipping

### Phase 4: Architecture Tuning (2-3 days)
- [ ] MDFA-specific: dilation rates, kernel sizes
- [ ] MSTCN-specific: number of scales, fusion mechanism
- [ ] Transformer: positional encoding, attention heads
- [ ] Ensemble of top-K models

---

## Experiment Log

### Baseline Run
- **Date**: TBD
- **Model**: TBD
- **Config**: TBD
- **Results**: TBD
- **Notes**: TBD

---

## Best Results Tracker

| Date | Model | RMSE_norm | Improvement | Config | Notes |
|------|-------|-----------|-------------|--------|-------|
| TBD  | TBD   | TBD       | Baseline    | TBD    | TBD   |

---

## Key Learnings

- TBD

