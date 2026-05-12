# Paper Outline (frozen — Phase 0 output)

**Title:** Predicting Remaining Useful Life in Commercial Turbofans: A Unified Deep-Learning Pipeline for Variable-Length N-CMAPSS Flights

**Audience:** Professional / academic (PHM, aerospace ML, deep-learning prognostics).
**Target length:** 8 pages + references (article-class XeLaTeX).
**Build:** `latexmk -xelatex -shell-escape main.tex`.

## Claims to defend

- **C1 — Sequence length:** Operationally short windows (~1k timesteps) outperform full-flight sequences (~20k) for RUL prediction on N-CMAPSS by a large margin (target evidence: ~58% RMSE improvement). Hypothesis: recent degradation patterns dominate signal; long histories add noise without aiding prediction.
- **C2 — Multi-scale temporal attention:** A multi-scale TCN with global-fusion attention (MSTCN) is competitive with or better than transformer / WaveNet / RNN baselines on N-CMAPSS at fixed compute, supporting the value of attention across multiple temporal dilations.
- **C3 — Asymmetric loss for safety-critical RUL:** Penalizing late predictions more heavily than early ones (asymmetric MSE, 2× late penalty) is the appropriate training objective for safety-critical RUL because it aligns the loss surface with the operational cost asymmetry of missed maintenance.

## Section list

1. **Abstract + Introduction** — `sections/abstract_intro.tex`
2. **Related Work** — `sections/related_work.tex`
3. **Methods** — `sections/methods.tex` (data, preprocessing, asymmetric loss, MSTCN architecture)
4. **Experiments** — `sections/experiments.tex` (setup, baselines, headline results)
5. **Ablations & Discussion** — `sections/ablations_discussion.tex` (C1/C2/C3 ablations)
6. **Conclusion & Limitations** — `sections/conclusion.tex`

## Numbering & evidence rules (binding for downstream phases)

- Every numerical claim in any `.tex` file must reference a row ID in `paper/state/canonical_results.md`.
- New citations must add a key to `paper/references.bib` (seeded by `paper/state/findings/references_seed.bib`).
- XeLaTeX-only: use `fontspec` / `unicode-math`; do not load `inputenc` or `fontenc`.
- Figures live in `paper/figures/` as PDF or PNG. Tables live in `paper/tables/` or inline.

## Default reviewer panel (Phase 5)

- Methodological reviewer
- Numbers reviewer
- Writing/clarity reviewer
- Adversarial reviewer (fresh-context)
