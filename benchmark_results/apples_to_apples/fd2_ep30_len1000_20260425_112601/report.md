# Apples-to-Apples Benchmark

| Rank | Model | Acc@10 | Acc@20 | RMSE | RMSE(norm,fixed) | R2 | Runtime (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | wavenet | 87.13% | 99.50% | 6.739 | 0.0539 | 0.8813 | 2.3 |
| 2 | mstcn | 88.12% | 99.01% | 6.495 | 0.0520 | 0.8897 | 3.1 |
| 3 | cnn_gru | 29.70% | 59.41% | 19.812 | 0.1585 | -0.0265 | 0.4 |

Best by Accuracy@20: `wavenet` at `99.50%`.
Paper-gap proxy using fixed denominator: `RMSE(norm,fixed)=0.0539`.
