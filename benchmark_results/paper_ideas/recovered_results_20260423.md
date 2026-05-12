# Recovered Paper-Idea Results

This summary combines:

- Existing repo benchmarks in `EXPERIMENTS.md` and `FINAL_RESULTS_COMPARISON.md`
- Recovered offline W&B summaries from completed local runs
- The new reproducible harness in `scripts/benchmark_paper_ideas.py`

Important: these results were not all produced under one identical fresh sweep in this session. The uniform 10-way rerun was started but proved too expensive at 1000-step inputs for some models. The numbers below are the best completed local metrics available for each implemented idea.

## Accuracy@20 Ranking

| Rank | Idea / Model | Accuracy@20 | RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| 1 | CATA-TCN | 99.71% | 7.38 | Best pure `Accuracy@20` from local benchmark history |
| 2 | TTSNet | 99.41% | 8.15 | Very strong threshold accuracy, weaker RMSE |
| 3 | MSTCN (100-epoch run) | 99.12% | 7.04 | Strong balance, from `FINAL_RESULTS_COMPARISON.md` |
| 4 | CNN-GRU + asymmetric loss | 99.10% | 6.44 | Best overall RMSE among high-accuracy models |
| 5 | WaveNet + asymmetric loss | 98.50% | 6.73 | Strong alternative |
| 6 | Transformer + asymmetric loss | 98.50% | 6.75 | Strong alternative |
| 7 | TCN | 97.70% | 8.36 | Simpler baseline, still competitive |
| 8 | Inception-LSTM | 96.20% | 10.11 | Multi-scale convolution idea helps versus weaker baselines |
| 9 | ResNet-LSTM | 95.90% | 10.38 | Residual CNN blocks help, but less than pure attention/TCN families |
| 10 | MDFA | 79.18% | 14.33 | Multi-scale fusion idea underperformed locally |

## Current-Code Reruns Completed In This Session

| Model | Epochs | Accuracy@20 | RMSE | Source |
| --- | ---: | ---: | ---: | --- |
| CNN-GRU baseline | 20 | 98.53% | 10.97 | recovered W&B run `cnn_gru_baseline_fd1_ep20` |
| Inception-LSTM | 20 | 60.41% | 20.74 | recovered W&B run `inception_lstm_blocks_fd1_ep20` |
| ResNet-LSTM | 20 | 69.79% | 15.65 | recovered W&B run `residual_lstm_blocks_fd1_ep20` |
| Transformer | 20 | 97.36% | 7.22 | recovered W&B run `self_attention_transformer_fd1_ep20` |

## Interpretation

- If the target metric is strictly `Accuracy@20`, the strongest idea in the currently available local results is **CATA-TCN**.
- If you care about `Accuracy@20` but do not want to give up RMSE, the strongest compromise is **CNN-GRU + asymmetric loss**.
- **MSTCN** is the strongest multi-scale temporal-convolution variant currently available in the repo and reaches `99.12%` `Accuracy@20` in the longer run.

## Source Mapping

| Idea family | Paper source used for inspiration |
| --- | --- |
| CNN-LSTM / CNN-GRU hybrid | https://www.researchgate.net/publication/363110710_An_enhanced_CNN-LSTM_remaining_useful_life_prediction_model_for_aircraft_engine_with_attention_mechanism |
| Inception / residual CNN blocks | https://papers.phmsociety.org/index.php/ijphm/article/view/3284 |
| Transformer / self-attention | https://www.sciencedirect.com/science/article/abs/pii/S0951832022005038 |
| Transformer under N-CMAPSS censoring | https://papers.phmsociety.org/index.php/ijphm/article/view/4260 |
| Multi-scale temporal conv / hybrid temporal models | https://link.springer.com/article/10.1007/s10845-025-02703-4 |
| Sensor dependency / heterogeneous modeling | https://arxiv.org/abs/2405.04336 |
| Uncertainty-aware RUL estimation | https://arxiv.org/abs/2104.03613 |
| Multi-task learning | https://www.sciencedirect.com/science/article/abs/pii/S0952197625037078 |
| Domain adaptation across flight profiles | https://arxiv.org/abs/2302.01704 |

