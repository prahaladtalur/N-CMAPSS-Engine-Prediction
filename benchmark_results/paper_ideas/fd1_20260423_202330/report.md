# Paper-Inspired Accuracy Benchmark

- Dataset split: N-CMAPSS FD1
- Epoch budget: 1
- Seed: 42
- Ranking metric: Accuracy@20 descending, then Accuracy@10 descending, then RMSE ascending

| Rank | Experiment | Model | Acc@10 | Acc@20 | RMSE | MAE | R2 | Runtime (min) | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | cnn_gru_baseline | cnn_gru | 18.77% | 30.50% | 40.4932 | 34.6853 | -2.5216 | 0.1 | ok |

## Sources

- `cnn_gru_baseline`: [An enhanced CNN-LSTM remaining useful life prediction model for aircraft engine with attention mechanism (PeerJ Computer Science, 2022)](https://www.researchgate.net/publication/363110710_An_enhanced_CNN-LSTM_remaining_useful_life_prediction_model_for_aircraft_engine_with_attention_mechanism)
  Note: Family proxy: repo uses CNN-GRU instead of CNN-LSTM-CBAM.

## Winner

`cnn_gru_baseline` with `cnn_gru` delivered the best headline result: Accuracy@20=30.50%, Accuracy@10=18.77%, RMSE=40.493.
