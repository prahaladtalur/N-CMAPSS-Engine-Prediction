# Review-Response Controlled Experiments

These runs address reviewer concerns about seed variance, sequence-length confounding, loss comparisons, and virtual-sensor use.

## Dataset

- N-CMAPSS fd=1 via `rul-datasets`.
- Reader split: default `rul-datasets` split, with the last 20% of original training units used for validation.
- Reader max RUL values: 65 cycles.
- Resolution values: 1 second(s).
- Feature sets evaluated: all, physical.
- `all` = 4 operating conditions + 14 physical sensors + 14 virtual sensors; `physical` = no virtual sensors.

## top_cluster

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cnn_gru | all, T=1000, asymmetric_mse | 3 | 17.168 | 8.801 | 64.13% | 0.2561 |
| mstcn | all, T=1000, asymmetric_mse | 3 | 7.890 | 0.526 | 97.75% | 0.8659 |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.868 | 0.299 | 98.24% | 0.8986 |

## window_sweep

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=100, asymmetric_mse | 3 | 7.098 | 0.375 | 97.95% | 0.8916 |
| wavenet | all, T=250, asymmetric_mse | 3 | 6.880 | 0.531 | 98.44% | 0.8980 |
| wavenet | all, T=500, asymmetric_mse | 3 | 6.690 | 0.173 | 98.24% | 0.9038 |

## loss_compare

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=1000, asymmetric_huber | 3 | 8.904 | 3.300 | 95.31% | 0.8141 |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.868 | 0.299 | 98.24% | 0.8986 |
| wavenet | all, T=1000, mse | 3 | 7.214 | 0.559 | 97.65% | 0.8878 |

## feature_compare

| Model | Setting | Seeds | RMSE mean | RMSE std | Acc@20 mean | R2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | all, T=1000, asymmetric_mse | 3 | 6.868 | 0.299 | 98.24% | 0.8986 |
| wavenet | physical, T=1000, asymmetric_mse | 3 | 7.224 | 1.251 | 99.12% | 0.8857 |

