# Evidence Map (Phase 2 synthesis)

This file maps each frozen claim from `paper/state/outline.md` to the supporting and
contradicting evidence in the Phase 1 findings. Drafting agents must cite evidence
through this map and via the row IDs in `paper/state/canonical_results.md`.

Citation conventions used below:
- `R0xx` = raw row IDs in `paper/state/findings/results.csv`.
- `T0x` = canonical paper row IDs defined in `paper/state/canonical_results.md`.
- `findings/<file>:Lxx` = line number in the named findings file.
- Bib keys (`saxena2008damage`, etc.) are defined in `findings/references_seed.bib`.

---

## Claim C1 — Short windows (~1k) outperform full-flight (~20k) sequences

> "Operationally short windows (~1k timesteps) outperform full-flight sequences
> (~20k) for RUL prediction on N-CMAPSS by a large margin (target evidence: ~58%
> RMSE improvement)."

### Supporting findings (internal)

- `results.csv` R001 (MSTCN, FD1, seq=1000, 30 ep) RMSE 6.80, R² 0.9006 vs
  R008 (TCN, FD1, seq=20294, 30 ep) RMSE 16.13, R² 0.4415 — the single
  cleanest 1k-vs-20k pair on the *same compute budget*.
- `results.csv` R001-R007 (top-7 models, all seq=1000) cluster at RMSE 6.80–8.15
  vs R013-R019 (LSTM/BiLSTM/BiGRU/Attention-LSTM, all seq=20294) all at RMSE
  ~22.2 with R² < 0. Aggregate gap is ~58% RMSE reduction
  (`results_notes.md:13-26`, `findings/architectures.md:736`).
- `results.csv` R022 (WaveNet, FD1, seq=1000, apples-to-apples) RMSE 6.523 and
  R026 (MSTCN, FD2, seq=1000, apples-to-apples) RMSE 6.495 — show 1k regime
  is competitive across two FDs with controlled config.
- `results.csv` R028-R032 (FD1, seq=250, 10 ep) RMSE 10.7–24.5 — show that
  going *too short* (250) also hurts, so the win is specifically at ~1k, not
  "shorter is monotonically better."
- `findings/methods.md:92` (`prepare_sequences()` truncates to **last** N
  timesteps) — implementation matches the "recent-degradation-dominates"
  hypothesis behind C1.

### Supporting findings (external corroboration)

- `solismartin2021stacked` (3rd PHM-2021 N-CMAPSS challenge) uses **161-cycle**
  windows on the same dataset and reports RMSE 6.24 / NASA score 0.64 val
  (`findings/related_work.md:37`). Direct precedent that sub-1k windows beat
  full-flight on N-CMAPSS.
- `lovberg2021variable` (1st place 2021 PHM N-CMAPSS challenge) explicitly
  designs for variable-length input rather than ingesting full flights
  (`findings/related_work.md:31`).
- `fan2024star` STAR transformer uses **32–64 step** input on C-MAPSS
  (`findings/related_work.md:50`).
- `zhou2021informer` motivates input distillation/compression rather than raw
  long sequences (`findings/related_work.md:46`).

### Contradicting / weakening findings

- **Confounding with architecture choice.** The 20k-sequence rows in
  `results.csv` (R008, R013-R019) are dominated by RNNs (LSTM, BiLSTM, GRU,
  BiGRU, Attention-LSTM, CNN-LSTM) and TCN. The 1k-sequence rows (R001-R007,
  R022-R027) are dominated by attention/multi-scale models. **No single model
  appears at both 1k and 20k in the dataset**, so the 1k-vs-20k delta is
  confounded with the architecture-vs-architecture delta. The "58% better"
  number documented in `CLAUDE.md` and `FINAL_ANALYSIS_REPORT.md` therefore
  conflates C1 and C2.
- **Sample at full length is sparse.** Only 7 rows in `results.csv` use
  seq=20294 (R008, R013, R014, R016, R017, R018, R019). All RNN-family rows
  could be failing for vanishing-gradient reasons orthogonal to sequence
  length.
- `findings/ablation_gaps.md:24` (P0 #1) explicitly flags the missing
  *single-architecture* sweep over {100, 250, 500, 1000, 2000, 5000, 20000}
  as the load-bearing experiment for C1. It does not exist.
- `findings/ablation_gaps.md:27` flags that the 1k-vs-full comparison has
  **never been run on FD3-FD7** — generalization beyond FD1 (and partial FD2)
  is unverified.

### Open gaps (cross-ref `ablation_gaps.md`)

- **P0** Single-model sequence-length sweep (`ablation_gaps.md:24`) — needed
  to disentangle C1 from C2.
- **P0** Multi-FD replication FD3/FD4/FD5 at {1k, full} (`ablation_gaps.md:27`).
- **P1** Per-model sensitivity heatmap and memory profile
  (`ablation_gaps.md:28-29`).
- **P2** Last-N vs random/first-N window ablation (`ablation_gaps.md:30`).

### Reframing recommendation

The literal "58% better" claim is unsafe as-stated because it conflates C1
with C2. Phase 3 drafters **should reframe C1 as one of**:

  (a) "Short windows (~1k cycles) are state-of-the-art on N-CMAPSS, matching
      independent precedent (Solís-Martín 161 cycles, STAR 32–64 cycles), and
      models trained on full flights underperform by a large margin in our
      benchmarks," with the 58% number reported as a *ceiling* across the
      benchmark, not a controlled ablation.

  (b) Run the P0 single-architecture sweep before defending the magnitude.

If the sweep is not run, drafters must avoid the headline "58% RMSE
improvement from sequence-length truncation alone" formulation and instead
report the model-mixed delta with a caveat.

### Defensibility verdict

**Moderate.** The directional claim (short windows match or beat full flights
on N-CMAPSS) is strongly supported by both internal benchmarks and three
independent prior works. The *magnitude* claim (58%) is currently confounded
with model choice and needs either reframing or the P0 sweep to be defended
rigorously.

---

## Claim C2 — Multi-scale TCN with global-fusion attention (MSTCN) is competitive with transformer / WaveNet / RNN baselines at fixed compute

> "A multi-scale TCN with global-fusion attention (MSTCN) is competitive with
> or better than transformer / WaveNet / RNN baselines on N-CMAPSS at fixed
> compute, supporting the value of attention across multiple temporal
> dilations."

### Supporting findings (internal)

- `results.csv` R001-R003 (FD1, seq=1000, 30 ep, matched config from
  `FINAL_ANALYSIS_REPORT.md`): MSTCN 6.80 / Transformer 6.82 / WaveNet 6.84
  RMSE — three-way tie within 0.04 cycles. Top-6 (R001-R006) all use
  attention or multi-scale mechanisms.
- `results.csv` R026 (FD2, seq=1000, apples-to-apples): MSTCN 6.495 RMSE,
  best on FD2 by RMSE; WaveNet (R025) 6.739, CNN-GRU (R027) collapses to
  19.81 — MSTCN generalizes across at least two FDs while CNN-GRU does not.
- `results.csv` R046-R050 (MSTCN multiseed, FD1, seq=1000, 30 ep): mean RMSE
  8.14 ± 0.74, best seed 7.52 (R049), worst 9.18 (R050). Establishes
  reproducibility envelope.
- `results.csv` R059 (production MSTCN, FD1, seq=1000) RMSE 6.37, R² 0.9128,
  Acc@20 99.12% — best single MSTCN result on file.
- `findings/architectures.md:463-521` documents MSTCN's Global Fusion
  Attention + multi-scale dilation design (rates [1,2,4,8]) as the novel
  contribution.
- `findings/methods.md:333-510` provides the architectural detail (4 parallel
  ResidualTCNBlock branches, channel + temporal + cross-scale attention,
  ~150k params).

### Supporting findings (external corroboration)

- `xu2024msattn` (RESS 2024, vol. 250) — strongest direct precedent: MS-TCN
  with self-attention + global-fusion-attention, validated SOTA on C-MAPSS
  (`findings/related_work.md:62`).
- `fan2024star` STAR (Sensors 2024) — RMSE 10.61 on FD001, demonstrates
  attention-augmented hybrids dominate the recent SOTA frontier
  (`findings/related_work.md:50`).
- `ttsnet2025` — transformer + TCN + self-attention fusion, RMSE 11.02 on
  C-MAPSS FD001 (`findings/related_work.md:53`).
- `bai2018tcn` — foundational TCN reference for the backbone
  (`findings/related_work.md:60`).

### Contradicting / weakening findings

- **Critical tension: apples-to-apples FD1 ranks MSTCN 3/5, not 1.** In the
  controlled apples-to-apples benchmark
  (`benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/`),
  MSTCN R024 RMSE 7.604 (PHM 1.0033, Acc@20 98.53%) is **behind**
  WaveNet R022 RMSE 6.523 (PHM 0.7256, Acc@20 98.83%) and CNN-GRU R023
  RMSE 7.006 (PHM 0.8396, Acc@20 98.53%). The 6.80 number from R001 / 
  `FINAL_ANALYSIS_REPORT.md` is a *single best run* under different control,
  not a head-to-head winner. `findings/ablation_gaps.md:40` flags this
  explicitly. **This is a real tension that must be surfaced in the paper.**
- **No iso-parameter comparison.** `findings/ablation_gaps.md:52` (P0 #4)
  notes that MSTCN (~150k params, `methods.md:504`) has not been compared to
  Transformer/WaveNet at matched parameter count. The "fixed compute" half of
  C2 is therefore unverified.
- **Multiseed variance is non-trivial.** R046-R050 show MSTCN spans
  RMSE 7.52–9.18 across seeds 42-44; the gap to WaveNet's apples-to-apples
  6.523 is ≥1 RMSE for two of three MSTCN seeds.
- **No single-scale ablation.** `findings/ablation_gaps.md:50` (P0 #3a) — the
  "multi-scale matters" sub-claim is unverified because we have not trained
  MSTCN with one dilation rate vs many.
- **No GFA ablation.** `findings/ablation_gaps.md:51` (P0 #3b) — the
  "global fusion attention matters" sub-claim is unverified.
- **MDFA in our impl underperforms its own paper.** R007 (MDFA) RMSE 14.33,
  R² 0.5588 on the same FD1/1k/30ep grid where MSTCN wins (`results.csv`,
  `findings/architectures.md:706-712`) — suggests our SOTA implementations
  may be tuning-sensitive in ways that affect cross-paper comparability.

### Open gaps (cross-ref `ablation_gaps.md`)

- **P0** Single-scale vs multi-scale MSTCN ablation (`ablation_gaps.md:50`).
- **P0** Remove-GFA ablation (`ablation_gaps.md:51`).
- **P0** Iso-parameter MSTCN/Transformer/WaveNet comparison
  (`ablation_gaps.md:52`).
- **P1** Cross-FD3-FD5 generalization for the top-3 (`ablation_gaps.md:53`).
- **P1** Remove pre-TCN self-attention ablation (`ablation_gaps.md:54`).
- **P2** Channel/temporal/cross-scale attention sub-ablation in GFA, latency
  comparison (`ablation_gaps.md:55-56`).

### Reframing recommendation

The "MSTCN is the winner" framing in `CLAUDE.md` / `FINAL_ANALYSIS_REPORT.md`
is **not safely defensible** as a headline claim. Phase 3 drafters should
reframe C2 as one of:

  (a) "Multi-scale temporal attention models (MSTCN, Transformer, WaveNet)
      form a tight performance cluster on FD1 (RMSE 6.5–7.6 across our
      benchmarks), substantially ahead of single-scale RNN baselines
      (RMSE >22). MSTCN is among the strongest in this cluster and best on
      FD2 RMSE, with WaveNet best on FD1 in the apples-to-apples
      benchmark."

  (b) Restrict the headline to "attention-augmented multi-scale architectures
      dominate single-scale RNN baselines" and treat MSTCN-vs-WaveNet as a
      reporting nuance.

Either way: report **both** R001-R003 (best-run leaderboard) **and**
R022-R024 (apples-to-apples) and acknowledge the rank-flip explicitly.

### Defensibility verdict

**Moderate (leaning weak as currently stated).** The "MSTCN beats baselines"
framing is contradicted by the most rigorous internal benchmark we have. The
weaker, defensible framing — "MSTCN is competitive with the strongest
attention/multi-scale baselines and decisively beats RNN baselines" — is
strongly supported. The headline should be reframed accordingly, or the P0
ablations and iso-parameter sweep need to run.

---

## Claim C3 — Asymmetric MSE (2× late penalty) is the appropriate training objective for safety-critical RUL

> "Penalizing late predictions more heavily than early ones (asymmetric MSE,
> 2× late penalty) is the appropriate training objective for safety-critical
> RUL because it aligns the loss surface with the operational cost asymmetry
> of missed maintenance."

### Supporting findings (internal)

- `findings/methods.md:120-156` defines asymmetric_mse(α=2.0) and gives the
  safety rationale: late predictions risk in-flight failure, early
  predictions are conservative.
- `findings/architectures.md:774-806` documents that all 22 registered models
  default to `asymmetric_mse(alpha=2.0)`.
- `findings/methods.md:556-591` documents the PHM08 score (Saxena 2008)
  asymmetric structure (exp(d/10) late penalty vs exp(-d/13) early penalty,
  ~3× asymmetry) — the loss family is consistent with the official metric.
- `results.csv` R059 (production MSTCN, asymmetric_mse) Acc@20 99.12%, PHM
  0.7327 — the production-grade safety metric is high.
- `results.csv` R022-R026 apples-to-apples runs (all asymmetric_mse) report
  PHM scores in the 0.58–1.00 range; comparable to literature PHM scores.

### Supporting findings (external corroboration)

- `saxena2008damage` — foundational asymmetric exponential PHM scoring
  function; the canonical justification for asymmetric losses in turbofan
  prognostics (`findings/related_work.md:15`). Anchors the conceptual case
  for C3.
- `mpm2024loss` (RESS 2025) — "improved multiple-penalty-mechanism loss"
  pairs RMSE with similarity-based late-prediction penalties; reports
  significantly improved advanced-prediction probability vs plain RMSE on
  C-MAPSS. Most recent and most direct prior work
  (`findings/related_work.md:71`).

### Contradicting / weakening findings

- **No symmetric MSE baseline exists in our benchmarks.**
  `findings/ablation_gaps.md:66, 76` — every benchmark in `results.csv`
  uses asymmetric_mse or asymmetric_huber. **There is no controlled A/B
  comparison of asymmetric vs symmetric loss with our models.** The "8–15%
  PHM improvement" cited in `EXPERIMENTS.md` is an old archive number, not
  reproduced in current benchmarks (`ablation_gaps.md:68`).
- **No α-sensitivity sweep.** `findings/ablation_gaps.md:77` — α=2.0 is the
  default but has not been justified against α ∈ {1.0, 1.5, 3.0}. The choice
  is essentially arbitrary in the current evidence.
- **Internal contradiction: the BEST_ACCURACY_RECIPE actually uses
  asymmetric_huber, not asymmetric_mse.** `findings/methods.md:326`,
  `findings/ablation_gaps.md:70`, `train_model.py:488-489`. The optimization
  recipe contradicts the loss claim. R054 (asymmetric_huber, 100 ep)
  achieves Acc@20 93.84% but RMSE 9.13 — possibly worse than asymmetric_mse
  on RMSE, better on smoothness.
- **PHM-score-as-loss is not tested.** Drafters can argue asymmetric_mse
  approximates the PHM scoring asymmetry, but no experiment confirms that
  training with asymmetric_mse correlates with low test-time PHM
  (`ablation_gaps.md:81`).

### Open gaps (cross-ref `ablation_gaps.md`)

- **P0** Symmetric MSE vs asymmetric_mse(α=2) head-to-head on top-3 models
  (`ablation_gaps.md:76`).
- **P0** α ∈ {1.0, 1.5, 2.0, 3.0} sweep (`ablation_gaps.md:77`).
- **P0** asymmetric_mse vs asymmetric_huber A/B (`ablation_gaps.md:78`).
- **P1** Cross-FD loss generalization, operational-cost-derivation of α,
  PHM-score-as-evaluation correlation (`ablation_gaps.md:79-81`).
- **P2** Sample-weighting × loss interaction, signed-error breakdown
  (`ablation_gaps.md:82-83`).

### Reframing recommendation

C3's "appropriate" claim is currently a design argument, not an empirical
result, in our work. Drafters should either:

  (a) Reframe C3 as "we adopt asymmetric MSE (α=2) as the training
      objective, motivated by the asymmetric PHM scoring function (Saxena
      2008) and recent prior work (`mpm2024loss`); a controlled comparison
      against symmetric MSE on our top models is left to future work."
      This is honest and defensible.

  (b) Run the P0 symmetric-MSE comparison and α-sensitivity sweep before
      the headline claim is made.

Drafters must **not** quote the "8–15% PHM improvement" number from
`EXPERIMENTS.md` without rerunning it (the source is stale per
`ablation_gaps.md:68`).

The α=2.0-vs-asymmetric_huber-α=1.5 internal contradiction must also be
resolved or acknowledged in the paper — pick one as the primary loss for
the headline claim and explain when the other is used.

### Defensibility verdict

**Weak as a headline empirical claim; moderate as a methodological design
choice.** Without a symmetric-MSE baseline or an α-sensitivity sweep, the
"asymmetric is appropriate" claim rests entirely on (i) the conceptual
alignment with Saxena 2008 PHM scoring and (ii) external prior work. As a
*choice* it is defensible; as an *empirical contribution* of this paper it
is not currently supported. Recommend reframing toward (a) above, or
running the P0 ablations.

---

## Cross-claim summary

| Claim | Internal support | External support | P0 gaps | Verdict |
|-------|------------------|------------------|---------|---------|
| C1 (sequence length) | Strong directional, weak controlled | Strong (3 PHM-2021 + STAR + Informer) | 2 (single-model sweep, multi-FD) | Moderate |
| C2 (MSTCN competitive) | Mixed (best-run wins, apples-to-apples 3rd) | Strong (Xu2024, STAR, TTSNet) | 3 (single-scale, no-GFA, iso-param) | Moderate (currently overstated) |
| C3 (asymmetric loss) | None (no baseline) + internal contradiction (Huber recipe) | Moderate (Saxena2008, MPM2024) | 3 (sym-MSE A/B, α sweep, MSE-vs-Huber) | Weak as empirical claim |

True blockers (vs gaps) are listed in `paper/state/blockers.md`.
