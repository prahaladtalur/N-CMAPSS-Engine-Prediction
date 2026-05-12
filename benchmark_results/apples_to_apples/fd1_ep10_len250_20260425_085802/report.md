# Apples-to-Apples Benchmark

| Rank | Model | Acc@10 | Acc@20 | RMSE | RMSE(norm,fixed) | R2 | Runtime (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | wavenet | 55.72% | 96.48% | 11.169 | 0.0894 | 0.7321 | 0.4 |
| 2 | mstcn | 67.16% | 91.50% | 10.748 | 0.0860 | 0.7519 | 0.4 |
| 3 | cata_tcn | 39.88% | 75.66% | 16.651 | 0.1332 | 0.4045 | 0.2 |
| 4 | transformer | 37.83% | 69.21% | 15.817 | 0.1265 | 0.4627 | 0.5 |
| 5 | cnn_gru | 23.46% | 46.92% | 24.502 | 0.1960 | -0.2894 | 0.1 |

Best by Accuracy@20: `wavenet` at `96.48%`.
Paper-gap proxy using fixed denominator: `RMSE(norm,fixed)=0.0894`.
