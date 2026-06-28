# SOTA-Chase Best Single Run

This sweep was run to chase a leaderboard-style best number, not to replace the
three-seed review-response result.

Best completed row:

| Model | FD | Feature set | Window | Loss | Seed | RMSE | MAE | R2 | Accuracy@20 |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| WaveNet | 1 | all | 1000 | asymmetric_mse | 47 | 6.197 | 4.803 | 0.9175 | 99.71 |

Caveat: this is a best-seed result from a partially stopped sweep. It is useful
as a repo leaderboard number, but it should not be presented as a robust SOTA
claim unless the comparison protocol is matched.
