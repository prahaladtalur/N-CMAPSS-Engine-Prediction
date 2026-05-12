# Related Work — Literature Review

Scope: literature directly relevant to the three claims of the paper.

- **C1**: Shorter input windows (~1k cycles) outperform full ~20k flight sequences on N-CMAPSS.
- **C2**: Multi-scale temporal attention (e.g., MSTCN-style) is competitive with transformer / RNN baselines.
- **C3**: Asymmetric MSE (late-prediction-penalizing) loss is appropriate for safety-critical RUL.

The review groups 11 papers by method family. Numerical metrics are reproduced as reported by each paper; we mark `—` when a number is not reported in a directly comparable form. Bib keys correspond to entries in `references_seed.bib`.

---

## 1. Datasets and benchmarks (foundational)

### 1.1 Saxena et al. (2008) — original C-MAPSS dataset and asymmetric scoring function
[`saxena2008damage`] introduces the C-MAPSS turbofan run-to-failure simulator and, critically, defines the **asymmetric exponential PHM score** in which late predictions are penalized exponentially harder than early ones (penalty parameters 13 vs. 10 in the 2008 challenge). This is the canonical justification for asymmetric losses in turbofan prognostics and is the primary citation for our **C3**. The four-subset (FD001–FD004) protocol it established remains the dominant pre-N-CMAPSS benchmark.

### 1.2 Arias-Chao et al. (2021) — N-CMAPSS dataset paper
[`ariaschao2021ncmapss`] introduces the new C-MAPSS (N-CMAPSS) dataset with eight subsets (DS01–DS08) of run-to-failure trajectories generated under real recorded flight conditions. Flights span ascent–cruise–descent with sample-rate-level data, producing trajectories on the order of 10⁴–10⁵ time steps per unit (the "20k full-flight" regime our paper targets). This is the foundational citation for everything that follows on N-CMAPSS.

### 1.3 Arias-Chao et al. (2022) — physics-augmented hybrid baseline on N-CMAPSS
[`ariaschao2022fusing`] (Reliability Engineering & System Safety) fuses physics-based calibration of unobservable health parameters with a deep regressor and reports it extends the prediction horizon ~127% over a purely data-driven baseline on a fleet of nine N-CMAPSS engines. We cite it as the canonical hybrid-baseline benchmark on N-CMAPSS.

---

## 2. PHM 2021 N-CMAPSS Data Challenge submissions (directly comparable)

Three published submissions to the 2021 PHM Society Data Challenge — all working on the same N-CMAPSS subset — collectively form the strongest apples-to-apples comparison set for our paper.

### 2.1 Lövberg (2021) — variable-length CNN with denoising normalization (most relevant to C1)
[`lovberg2021variable`] proposes a deep dilated-convolutional network that explicitly accepts **variable-length** input sequences, plus a learned normalization that maps flight conditions → expected sensor outputs and treats residuals as degradation features. The variable-length design is conceptually closest to our paper's flight-cycle handling and is the strongest prior art for our **C1** (sequence-length sensitivity). Won the 2021 PHM Society Data Challenge.

### 2.2 DeVol, Saldana & Fu (2021) — Inception-style CNN, 2nd place
[`devol2021inception`] uses inception modules so a single CNN can capture features at multiple temporal receptive fields from a window of N-CMAPSS sensor data. This is a multi-scale-CNN baseline directly relevant to our **C2**. Specific window length is not stated in the abstract; the architecture is window-based rather than full-flight.

### 2.3 Solís-Martín, Galán-Páez & Borrego-Díaz (2021) — stacked DCNN, 3rd place
[`solismartin2021stacked`] uses a two-level stacked DCNN: Level-1 encoder windows of **L_w = 161 cycles**, with Level-2 ingesting sparse encodings (step ≈ 989 s). Reported Level-2 RMSE = 6.24, NASA score = 0.64 on validation (test score 3.651). The 161-cycle window — far shorter than the full ~20k-step flight — is **direct corroboration of our C1**.

---

## 3. Attention- and transformer-based prognostics (relevant to C2)

### 3.1 Vaswani et al. (2017) — Transformer
[`vaswani2017attention`] is the canonical reference for self-attention as a sequence-modeling primitive; cited as the architectural origin of every transformer-based RUL model below.

### 3.2 Zhou et al. (2021) — Informer for long-sequence forecasting
[`zhou2021informer`] (AAAI'21 best paper) introduces ProbSparse self-attention with O(L log L) complexity and self-attention distilling for long sequences. Relevant to C1 because it demonstrates that even efficient long-sequence transformers benefit from input compression / distillation rather than raw long inputs.

### 3.3 Fan, Li & Chang (2024) — STAR: two-stage hierarchical transformer
[`fan2024star`] (Sensors) proposes the STAR framework with sequential **temporal** and **sensor-wise** attention, plus a hierarchical encoder–decoder for multi-scale predictions. Reports RMSE 10.61 / 13.47 / 10.71 / 15.87 (Score 169 / 784 / 202 / 1449) on FD001–FD004. Uses **input length 32–64 timesteps** — again a short-window choice (C1) by a state-of-the-art attention model.

### 3.4 TTSNet (2025) — Transformer + TCN + self-attention fusion
[`ttsnet2025`] (Sensors, MDPI) fuses three parallel branches (transformer, TCN, multi-head self-attention) over a noise-smoothed input. Reports RMSE 11.02 / 13.25 / 11.06 / 18.26 (Score 194.6 / 874.1 / 200.1 / 1968.5) on FD001–FD004. A direct C2 datapoint that combining attention with TCN (as our MSTCN-style claim asserts) is competitive at the SOTA frontier.

---

## 4. Multi-scale TCN methods (most relevant to C2)

### 4.1 Bai, Kolter & Koltun (2018) — generic TCN
[`bai2018tcn`] is the canonical reference for the dilated-causal-convolution TCN block; it shows TCNs match or beat LSTMs on standard sequence-modeling benchmarks while training faster and exhibiting longer effective memory. Foundational citation for our MSTCN backbone.

### 4.2 Xu, Zhang & Miao (2024) — attention-based multi-scale TCN
[`xu2024msattn`] (Reliability Engineering & System Safety, vol. 250, art. 110288) proposes an end-to-end **MSTCN with Self-Attention at the head and Global Fusion Attention at the tail**, validated on C-MAPSS. The architecture is essentially the same family our paper places in the "MSTCN" slot, and is the strongest direct evidence for **C2**.

---

## 5. Asymmetric / cost-sensitive RUL losses (relevant to C3)

### 5.1 Saxena et al. (2008) PHM score — see §1.1 above; the foundational asymmetric loss in this domain.

### 5.2 Improved multiple-penalty-mechanism (MPM) loss — Reliability Engineering & System Safety (2025)
[`mpm2024loss`] proposes a **multiple penalty mechanism loss** that pairs RMSE with similarity-based penalties tuned to suppress *late-prediction lag*; reports significantly higher advanced-prediction probability than plain RMSE on C-MAPSS subsets. Same authoring community as Xu2024 / RESS. This is the most recent and most direct prior work on the asymmetric-loss design space our **C3** occupies.

---

## 6. N-CMAPSS data-driven baselines (numeric anchors for our results table)

### 6.1 ANN-Flux (referenced through the survey hits, listed in "Fault Prognosis of Turbofan Engines" arXiv 2303.12982)
[`ncmapss_eventual2023`] (Khan et al., 2023, arXiv:2303.12982) reports an ANN-Flux baseline on N-CMAPSS achieving **RMSE 7.75** and NASA score **4.34** — a 38% / 42% improvement over earlier published N-CMAPSS results (RMSE 12.50, score 7.50). We cite this as a representative N-CMAPSS RMSE/score baseline outside the PHM challenge.

---

## Synthesis with respect to our claims

**C1 (sequence length).** Two of the three top PHM-2021 N-CMAPSS challenge submissions explicitly choose short windows: Solís-Martín uses **161-cycle** windows; Lövberg uses dilated-CNNs designed precisely to *handle* but not require full-length input. STAR (Fan 2024) uses **32–64-step** input on C-MAPSS. We therefore have strong precedent that sub-thousand-cycle windows are state of the art; our contribution is to *quantify* the gap between ~1k-cycle and full ~20k-cycle inputs on N-CMAPSS in a controlled comparison.

**C2 (multi-scale temporal attention).** Xu et al. (2024) is the closest prior art and validates MSTCN-with-attention on C-MAPSS. STAR and TTSNet show that attention-augmented hybrids dominate the recent SOTA frontier. Our novelty is to evaluate this family on the variable-length, multi-condition N-CMAPSS regime against transformer and TCN-only baselines under matched input length.

**C3 (asymmetric loss).** Saxena 2008 is the foundational reference; the 2025 MPM-loss paper [`mpm2024loss`] is the most recent direct prior work. Our 2× late-prediction-penalty asymmetric MSE is a simpler instantiation and serves as a credible default; we cite both in support.

---

## Comparison Table

Subset/dataset and metrics are reproduced as reported by each paper; "—" = not reported in a directly comparable form. "Win." = window/input length used (cycles or timesteps as reported).

| Paper | bib_key | Method family | Dataset / Subset | RMSE | R² | PHM Score | Win. |
|---|---|---|---|---|---|---|---|
| Saxena et al. 2008 | `saxena2008damage` | Dataset + scoring | C-MAPSS FD001–FD004 | — | — | def. used by all later papers | n/a |
| Arias-Chao et al. 2021 | `ariaschao2021ncmapss` | Dataset | N-CMAPSS DS01–DS08 | — | — | — | n/a |
| Arias-Chao et al. 2022 | `ariaschao2022fusing` | Hybrid physics + DL | N-CMAPSS (9 engines) | — | — | +127% horizon vs DL | — |
| Lövberg 2021 (1st PHM21) | `lovberg2021variable` | Dilated CNN, var-length | N-CMAPSS (PHM21) | — | — | (challenge winner) | variable |
| DeVol et al. 2021 (2nd) | `devol2021inception` | Inception-CNN | N-CMAPSS (PHM21) | — | — | (2nd place) | window |
| Solís-Martín et al. 2021 (3rd) | `solismartin2021stacked` | Stacked DCNN | N-CMAPSS (PHM21) | 6.24 (val L2) | — | 0.64 val / 3.651 test | **161** |
| Vaswani et al. 2017 | `vaswani2017attention` | Transformer (foundational) | n/a (NLP) | — | — | — | n/a |
| Zhou et al. 2021 Informer | `zhou2021informer` | ProbSparse transformer | ETT/etc. (forecasting) | — | — | — | long-seq |
| Fan et al. 2024 STAR | `fan2024star` | 2-stage hier. transformer | C-MAPSS FD001/2/3/4 | 10.61 / 13.47 / 10.71 / 15.87 | — | 169 / 784 / 202 / 1449 | **32–64** |
| TTSNet 2025 | `ttsnet2025` | Transformer + TCN + SA | C-MAPSS FD001/2/3/4 | 11.02 / 13.25 / 11.06 / 18.26 | — | 194.6 / 874.1 / 200.1 / 1968.5 | window |
| Bai et al. 2018 | `bai2018tcn` | Generic TCN (foundational) | n/a | — | — | — | n/a |
| Xu, Zhang, Miao 2024 (MSTCN+SA+GFA) | `xu2024msattn` | MS-TCN + attention | C-MAPSS | (SOTA on all subsets, see paper) | — | (SOTA) | window |
| MPM loss 2025 | `mpm2024loss` | Asymmetric loss | C-MAPSS | — | — | (lower lag prob.) | n/a |
| Khan et al. 2023 (ANN-Flux) | `ncmapss_eventual2023` | ANN-Flux | N-CMAPSS | 7.75 | — | 4.34 (NASA score) | window |
