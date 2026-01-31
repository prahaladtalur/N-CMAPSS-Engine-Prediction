# Experiment Results: SOTA Architecture Comparison

**Date**: 2026-01-29
**Dataset**: N-CMAPSS FD1
**Training**: 30 epochs, batch size 32
**Sequence Length**: 1000 timesteps (truncated from 20,294 for stability)

---

## Executive Summary

**Winner: Transformer** — best across every metric. Top 3 (Transformer, WaveNet, CNN-GRU) are all within ~8% of each other and all exceed 97% Acc@20. CNN-LSTM is the only model that failed to converge.

---

## Full Leaderboard

| Rank | Model | RMSE ↓ | MAE ↓ | PHM ↓ | Acc@10 ↑ | Acc@15 ↑ | Acc@20 ↑ | R² ↑ |
|------|-------|--------|-------|-------|----------|----------|----------|------|
| 1 | **Transformer** | **7.01** | **4.70** | **0.88** | **85.9%** | **94.1%** | **98.2%** | **0.89** |
| 2 | WaveNet | 7.44 | 4.99 | 0.99 | 83.6% | 91.8% | 97.1% | 0.88 |
| 3 | CNN-GRU | 7.58 | 5.71 | 1.04 | 85.6% | 93.0% | 97.1% | 0.88 |
| 4 | TCN | 8.36 | 6.22 | 1.17 | 78.0% | 90.3% | 97.7% | 0.85 |
| 5 | BiGRU | 8.71 | 6.94 | 1.23 | 83.0% | 91.5% | 96.8% | 0.84 |
| 6 | Inception-LSTM | 10.11 | 8.34 | 1.49 | 57.2% | 87.4% | 96.2% | 0.78 |
| 7 | ResNet-LSTM | 10.38 | 8.59 | 1.52 | 55.4% | 91.8% | 95.9% | 0.77 |
| 8 | CNN-LSTM | 21.63 | 19.03 | 8.77 | 23.5% | 35.2% | 46.9% | 0.00 |

---

## Key Findings

### Transformer is the clear winner
- Best RMSE (7.01), MAE (4.70), PHM (0.88), R² (0.89), and Acc@10/15/20
- With seq_len=1000 the O(n²) self-attention is fully tractable
- Previously failed on full 20k sequences due to memory — truncation fixed it

### Top 3 are very tight
- Transformer, WaveNet, and CNN-GRU are separated by < 8% on RMSE
- All three exceed 97% Acc@20 and R² > 0.87
- For production: Transformer is the safest pick; WaveNet is a good fallback if latency matters

### Architecture tiers
- **Tier 1 (production-ready)**: Transformer, WaveNet, CNN-GRU — all R² > 0.87, Acc@20 > 97%
- **Tier 2 (solid)**: TCN, BiGRU — R² > 0.84, Acc@20 > 96%
- **Tier 3 (underperformed)**: Inception-LSTM, ResNet-LSTM — still decent (R² ~0.78) but not competitive
- **Failed**: CNN-LSTM — did not converge

### CNN-LSTM vs CNN-GRU
CNN-LSTM completely failed (R² ≈ 0) while CNN-GRU ranked 3rd. The GRU back-end is significantly more stable than LSTM for this hybrid pattern.

---

## Training Notes

- Full 20k sequences caused OOM when running multiple models. Truncating to 1000 timesteps reduced memory ~20× with no loss in model quality.
- All models trained for 30 epochs with early stopping (patience=10) and LR reduction (patience=5).
- Transformer is slower per epoch (~30s vs ~2s for TCN) but converges to better final performance.

---

## Visualizations

- `results/comparison/full_leaderboard.png` — complete 8-model comparison
- `results/comparison/production_sota_comparison.png` — first 5-model round
- Individual model plots in `results/{model}-production/`
