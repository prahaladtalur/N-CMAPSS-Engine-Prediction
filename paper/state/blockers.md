# Phase 2 Blockers

No blockers; gaps captured in `evidence_map.md` and numerical contradictions
resolved in `canonical_results.md`.

The most significant tensions surfaced during synthesis (apples-to-apples FD1
ranking MSTCN 3/5, no symmetric-MSE baseline, BEST_ACCURACY_RECIPE using
asymmetric_huber rather than asymmetric_mse, sequence-length effect
confounded with model choice) are all **reframings** of the three claims, not
blockers. They are documented inline in `evidence_map.md` with explicit
"Reframing recommendation" subsections so Phase 3 drafters can write the
paper honestly without new experiments.

If Phase 3 drafters disagree with a reframing — for example, if they want to
keep the headline "58% RMSE improvement from sequence length alone" or "MSTCN
is the winner" formulations — they must escalate back to Phase 2 and the
P0 ablations in `ablation_gaps.md` will become true blockers.
