# Apples-to-Apples Benchmark

| Rank | Model | Acc@10 | Acc@20 | RMSE | RMSE(norm,fixed) | R2 | Runtime (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | wavenet | 87.39% | 98.83% | 6.523 | 0.0522 | 0.9086 | 2.3 |
| 2 | cnn_gru | 88.86% | 98.53% | 7.006 | 0.0560 | 0.8946 | 0.7 |
| 3 | mstcn | 82.11% | 98.53% | 7.604 | 0.0608 | 0.8758 | 2.4 |

Best by Accuracy@20: `wavenet` at `98.83%`.
Paper-gap proxy using fixed denominator: `RMSE(norm,fixed)=0.0522`.
