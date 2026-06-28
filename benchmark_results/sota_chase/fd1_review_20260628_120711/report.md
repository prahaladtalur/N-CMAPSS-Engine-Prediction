# Review-Response Controlled Experiments

These runs address reviewer concerns about seed variance, sequence-length confounding, loss comparisons, and virtual-sensor use.

## Dataset

- N-CMAPSS fd=1 via `rul-datasets`.
- Reader split: default `rul-datasets` split, with the last 20% of original training units used for validation.
- Reader max RUL values: 65 cycles.
- Resolution values: 1 second(s).
- Feature sets evaluated: all.
- `all` = 4 operating conditions + 14 physical sensors + 14 virtual sensors; `physical` = no virtual sensors.

## top_cluster

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.661 | 0.564 | 98.83% | 0.9043 |

## window_sweep

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=250, asymmetric_mse | 3 | 7.099 | 0.732 | 97.85% | 0.8910 |
| wavenet | all, T=500, asymmetric_mse | 3 | 7.256 | 0.869 | 97.46% | 0.8858 |

## loss_compare

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.661 | 0.564 | 98.83% | 0.9043 |

## feature_compare

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.661 | 0.564 | 98.83% | 0.9043 |

