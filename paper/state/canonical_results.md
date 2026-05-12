# Canonical Results Table (Phase 2 synthesis)

Single source of truth for numerical claims in the paper. Every `.tex` file
produced in Phase 3 must cite numbers from this file by `paper_row_id`
(`T1`, `T2`, ...). Do **not** cite raw `R0xx` IDs from `results.csv` in the
LaTeX; those are the underlying provenance and live only in the
`source_csv_row` column of this table.

## Selection rules

When `results.csv` and the documentation files (`CLAUDE.md`,
`FINAL_ANALYSIS_REPORT.md`, etc.) disagree, the following precedence is used:

1. Apples-to-apples controlled benchmark (`benchmark_results/apples_to_apples/...`)
   beats single-best historical runs.
2. Multiseed mean (when available) beats best single seed.
3. Most recent run beats older run with the same protocol.
4. Production checkpoint metrics beat tuning-sweep partial metrics.

Where this rule yields a number that *contradicts* the headline number in
`CLAUDE.md` / `FINAL_ANALYSIS_REPORT.md`, a footnote explains the choice.

---

## Table 1 — Headline N-CMAPSS results (this paper, FD1)

All FD1 rows below use seq_length=1000, batch_size=32, 30 epochs,
asymmetric_mse(α=2.0), Adam(lr=0.001), unless noted.

| paper_row_id | model | fd_subset | seq_length | RMSE | R² | PHM_score | accuracy_at_20 | source_csv_row | notes |
|---|---|---|---|---|---|---|---|---|---|
| T1  | WaveNet            | FD1 | 1000 | 6.523 | 0.9086 | 0.7256 | 98.83 | R022 | Apples-to-apples FD1, best RMSE in head-to-head [1] |
| T2  | CNN-GRU            | FD1 | 1000 | 7.006 | 0.8946 | 0.8396 | 98.53 | R023 | Apples-to-apples FD1 [1] |
| T3  | MSTCN              | FD1 | 1000 | 7.604 | 0.8758 | 1.0033 | 98.53 | R024 | Apples-to-apples FD1, ranks 3/5 here [1][2] |
| T4  | MSTCN (production) | FD1 | 1000 | 6.373 | 0.9128 | 0.7327 | 99.12 | R059 | Production checkpoint, single best MSTCN on file [3] |
| T5  | MSTCN (multiseed mean, n=3) | FD1 | 1000 | 8.140 | 0.8565 | 1.130 | 97.27 | R046-R050 | Mean over seeds 42/43/44; std RMSE ±0.74 [4] |
| T6  | WaveNet (multiseed mean, n=2) | FD1 | 1000 | 8.238 | 0.8498 | 1.224 | 96.92 | R051-R053 | Mean over seeds 45/46 [4] |
| T7  | Transformer        | FD1 | 1000 | 6.82  | 0.9002 | —      | —     | R002 | Best historical run, FINAL_ANALYSIS_REPORT.md [2] |
| T8  | LSTM               | FD1 | 20294 | 22.28 | -0.0657 | —     | —     | R018 | Full-flight RNN baseline [2] |
| T9  | TCN                | FD1 | 20294 | 16.13 | 0.4415 | —      | —     | R008 | Full-flight TCN baseline [2] |
| T10 | MLP                | FD1 | 1000 | 17.45 | 0.3461 | —      | —     | R010 | No-temporal baseline [2] |

## Table 2 — Cross-FD generalization (this paper, FD2)

| paper_row_id | model | fd_subset | seq_length | RMSE | R² | PHM_score | accuracy_at_20 | source_csv_row | notes |
|---|---|---|---|---|---|---|---|---|---|
| T11 | MSTCN   | FD2 | 1000 | 6.495  | 0.8897 | 0.5755 | 99.01 | R026 | Apples-to-apples FD2, best RMSE [1] |
| T12 | WaveNet | FD2 | 1000 | 6.739  | 0.8813 | 0.6756 | 99.50 | R025 | Apples-to-apples FD2, best Acc@20 [1] |
| T13 | CNN-GRU | FD2 | 1000 | 19.812 | -0.0265 | 5.1463 | 59.41 | R027 | Apples-to-apples FD2; CNN-GRU collapses on FD2 [1] |

## Table 3 — Sequence-length effect (paired evidence for C1)

This table backs claim C1. It is *not* a controlled single-model sweep; the
1k-vs-20k pairs use different model families (see C1 evidence map for the
caveat). The 250-step rows are included to show that going *too short* also
hurts.

| paper_row_id | model | fd_subset | seq_length | RMSE | R² | accuracy_at_20 | source_csv_row | notes |
|---|---|---|---|---|---|---|---|---|
| T14 | MSTCN  | FD1 | 250  | 10.748 | 0.7519 | 91.50 | R029 | 10 epochs, short window [5] |
| T15 | WaveNet| FD1 | 250  | 11.169 | 0.7321 | 96.48 | R028 | 10 epochs, short window [5] |
| T16 | CNN-GRU| FD1 | 250  | 24.502 | -0.2894 | 46.92 | R032 | 10 epochs, short window [5] |
| T17 | MSTCN  | FD1 | 1000 |  7.604 | 0.8758 | 98.53 | R024 | (= T3) |
| T18 | WaveNet| FD1 | 1000 |  6.523 | 0.9086 | 98.83 | R022 | (= T1) |
| T19 | TCN    | FD1 | 20294| 16.13  | 0.4415 | —     | R008 | Full-flight TCN [2] |
| T20 | LSTM   | FD1 | 20294| 22.28  | -0.0657| —     | R018 | Full-flight LSTM [2] |
| T21 | BiLSTM | FD1 | 20294| 22.20  | -0.0584| —     | R014 | Full-flight BiLSTM [2] |

## Table 4 — Loss-function evidence (for C3, partial)

| paper_row_id | model | fd_subset | seq_length | loss | RMSE | R² | accuracy_at_20 | source_csv_row | notes |
|---|---|---|---|---|---|---|---|---|---|
| T22 | MSTCN | FD1 | 1000 | asymmetric_mse(α=2.0)        | 7.604 | 0.8758 | 98.53 | R024 | (= T3) |
| T23 | MSTCN | FD1 | 1000 | asymmetric_huber(α=1.5,δ=0.08)| 9.132 | 0.8209 | 93.84 | R054 | High-accuracy recipe; 100 ep [6] |
| T24 | MSTCN | FD1 | 1000 | asymmetric_mse(α=2.0)        | 6.373 | 0.9128 | 99.12 | R059 | (= T4); production model [3] |

**Caveat:** No symmetric-MSE row exists in the current benchmarks. T22-T24
are not a controlled loss A/B; they are different runs. C3 should not be
defended as an empirical result from this table alone — see
`evidence_map.md` C3.

---

## Footnotes — resolution of numerical contradictions

**[1] Apples-to-apples primary source.** Rows
`benchmark_results/apples_to_apples/fd1_ep30_len1000_20260425_090057/results.json`
and `.../fd2_ep30_len1000_20260425_112601/results.json`. These runs use a
single benchmark harness (`scripts/benchmark_apples_to_apples.py`) with
identical hyperparameters across all five models, are date-stamped 2026-04-25
(most recent), and report PHM score and accuracy in addition to RMSE/R².
Selected as the canonical head-to-head numbers per selection rule 1.

**[2] FINAL_ANALYSIS_REPORT.md numbers retained as historical leaderboard.**
Rows R001-R019 in `results.csv` are extracted from `FINAL_ANALYSIS_REPORT.md`
and report RMSE 6.80 / 6.82 / 6.84 for MSTCN / Transformer / WaveNet at FD1,
seq=1000, 30 ep. **These contradict the apples-to-apples ranking** ([1])
where MSTCN ranks 3/5 at RMSE 7.604, behind WaveNet 6.523 and CNN-GRU 7.006.
Resolution: we report **both** sets of numbers (T1–T3 for the rigorous
head-to-head; T7 for the historical Transformer number where there is no
apples-to-apples Transformer entry on the same date), and the paper must
cite the apples-to-apples set as the primary "competitive at fixed compute"
evidence. The 6.80 number that appears in `CLAUDE.md` and
`FINAL_ANALYSIS_REPORT.md` as the "best model" headline is **not** repeated
as a headline in this paper; it is reported as the best historical single run
(R001), not as the apples-to-apples result.

**[3] Production checkpoint (R059).** Source:
`models/production/metrics.json`. RMSE 6.373 — better than any apples-to-apples
or multiseed result on file. We report it (T4, T24) as the best MSTCN single
configuration on disk, but **do not use it as the headline benchmark number**
for the paper because (a) it is a single seed with unspecified epoch count
in `results.csv` and (b) it is a development checkpoint, not the apples-to-apples
result. Headline MSTCN number for the paper is T3 (apples-to-apples) with T4
reported as "best observed."

**[4] Multiseed.** `results/mstcn_multiseed_summary.json` (seeds 42-44) and
`results/wavenet_multiseed_summary.json` (seeds 45-46). Computed as simple
mean across the available seeds in `results.csv`. WaveNet only has 2 seeds,
so the std is wide; flag this if reporting variance.

**[5] Short-window 10-epoch runs.** Source:
`benchmark_results/apples_to_apples/fd1_ep10_len250_20260425_085802/results.csv`.
These were run only for 10 epochs (vs 30 for T1–T3), so RMSE is inflated
relative to converged numbers. Report as "shows 250 is too short," not as a
direct head-to-head with the 1000-step rows.

**[6] BEST_ACCURACY_RECIPE / asymmetric_huber.** Source:
`models/mstcn_20260419_182347/metadata.json`, run with
`train_model.py:475-516` BEST_ACCURACY_RECIPE config. The recipe **uses
asymmetric_huber(α=1.5, δ=0.08), not asymmetric_mse(α=2.0)**, contradicting
the C3 claim. Reported here so drafters can decide how to handle this in the
paper (see `evidence_map.md` C3 reframing recommendation).

**[7] Numbers excluded from the canonical table.** The following rows in
`results.csv` are intentionally **not** promoted to canonical paper rows:

- R033, R041-R045 — 1-epoch tuning baselines, metrics not converged.
- R035-R040 — 75-epoch hyperparameter sweep, superseded by 30-epoch tuned
  runs and missing RMSE/MAE.
- R055-R058 — recovered offline W&B metrics, lower rigor than apples-to-apples.
- R034 — historical CNN-GRU "RMSE 6.44" from `README.md`; cannot be
  reproduced (single epoch, missing fields), so excluded per
  `results_notes.md:192-195`.

---

## Table 5 — External literature comparison (for the related-work section)

This table reproduces RMSE numbers from the related-work corpus
(`findings/related_work.md`) and is provided for the related-work and
discussion sections. **Most of these are C-MAPSS, not N-CMAPSS, so the
apples-to-apples comparison with our results is limited.** Drafters must
state this explicitly in any prose that uses Table 5.

| paper | bib_key | dataset | best RMSE | window | notes | direct_comparable_to_us? |
|---|---|---|---|---|---|---|
| Saxena et al. 2008          | `saxena2008damage`       | C-MAPSS FD001-004                     | —                                  | n/a       | Defines PHM scoring function used by all later work | No (dataset paper) |
| Arias-Chao et al. 2021      | `ariaschao2021ncmapss`   | N-CMAPSS DS01-08                      | —                                  | n/a       | N-CMAPSS dataset paper | No (dataset paper) |
| Arias-Chao et al. 2022      | `ariaschao2022fusing`    | N-CMAPSS (9 engines)                  | —                                  | —         | Hybrid physics+DL; +127% horizon vs DL | Yes — N-CMAPSS, but no RMSE reported in the form we use |
| Lövberg 2021 (1st PHM21)    | `lovberg2021variable`    | N-CMAPSS PHM-2021 challenge           | —                                  | variable  | Challenge winner; variable-length CNN; precedent for C1 | Yes — N-CMAPSS |
| DeVol et al. 2021 (2nd)     | `devol2021inception`     | N-CMAPSS PHM-2021                     | —                                  | window    | 2nd place; inception multi-scale CNN | Yes — N-CMAPSS |
| Solís-Martín et al. 2021    | `solismartin2021stacked` | N-CMAPSS PHM-2021                     | 6.24 (val L2), 3.651 (test NASA score) | **161 cycles** | 3rd place; explicit short-window precedent for C1 | Yes — N-CMAPSS |
| Khan et al. 2023 (ANN-Flux) | `ncmapss_eventual2023`   | N-CMAPSS                              | 7.75                               | window    | NASA score 4.34; 38% better than earlier published N-CMAPSS work | Yes — N-CMAPSS |
| Vaswani et al. 2017         | `vaswani2017attention`   | n/a                                   | —                                  | n/a       | Transformer foundational reference | No |
| Zhou et al. 2021 Informer   | `zhou2021informer`       | ETT etc.                              | —                                  | long-seq  | Long-sequence transformer; supports C1 motivation | No (different domain) |
| Fan et al. 2024 STAR        | `fan2024star`            | C-MAPSS FD001/2/3/4                   | 10.61 / 13.47 / 10.71 / 15.87      | **32-64** | Two-stage hierarchical transformer; supports C1 + C2 | **No (C-MAPSS, not N-CMAPSS)** |
| TTSNet 2025                 | `ttsnet2025`             | C-MAPSS FD001/2/3/4                   | 11.02 / 13.25 / 11.06 / 18.26      | window    | Transformer+TCN+SA fusion; supports C2 | **No (C-MAPSS)** |
| Bai et al. 2018             | `bai2018tcn`             | n/a                                   | —                                  | n/a       | TCN foundational reference | No |
| Xu, Zhang, Miao 2024        | `xu2024msattn`           | C-MAPSS                               | reported SOTA all subsets (see paper) | window | MSTCN+SA+GFA — **direct precedent for our MSTCN family**; supports C2 | **No (C-MAPSS, but architectural twin)** |
| MPM-loss 2025               | `mpm2024loss`            | C-MAPSS                               | —                                  | n/a       | Asymmetric/multi-penalty loss; supports C3 | No (C-MAPSS) |

**Comparability caveat (must appear in paper):** N-CMAPSS sequences are
~10⁴–10⁵ timesteps per cycle (`ariaschao2021ncmapss`, `findings/related_work.md:19`),
whereas C-MAPSS sequences are typically a few hundred. RMSE numbers across
the two datasets are not on the same scale. Direct head-to-head comparisons
in the paper are therefore restricted to the N-CMAPSS rows
(`solismartin2021stacked`, `lovberg2021variable`, `devol2021inception`,
`ncmapss_eventual2023`, `ariaschao2022fusing`).

The closest single literature anchor for "what is a good N-CMAPSS RMSE?" is
Solís-Martín 2021 (validation L2 RMSE 6.24, 161-cycle window) and Khan 2023
(RMSE 7.75). Our MSTCN apples-to-apples (T3, RMSE 7.604) and best WaveNet
apples-to-apples (T1, RMSE 6.523) sit within that band.
