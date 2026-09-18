# Response to Reviewers — Draft

**Revised manuscript:** *Beyond Cycle Count: Sensor-Based Maintenance Warnings for Simulated Turbofan Engines*

We thank the editor and reviewers for identifying that the original manuscript's evidence did not support its broad architecture framing. The revised manuscript remains part of the same N-CMAPSS engine-prognostics project, but it narrows the research question in direct response to that feedback. Instead of treating the experimental architecture registry as a completed benchmark, it now asks one operationally motivated question: whether measured engine telemetry adds final-window warning information beyond elapsed cycle and current operating conditions on held-out simulated engines.

The revision preserves the manuscript's central subject—predicting the approach to simulated turbofan-engine failure from N-CMAPSS data—while replacing claims that the reviewers found insufficiently supported. The earlier three-model ranking, the 22-model registry description, the selected best-seed result, and the unnormalized runtime comparison have been removed. The revised analysis uses 90 complete tiny-N-CMAPSS trajectories, explicit feature provenance, nested engine-held-out validation and threshold selection, complete-engine bootstrap intervals, and five-seed robustness checks. No label, engine identifier, auxiliary health-state field, or outer-test label is used as an input or for model and threshold selection.

## Reviewer 1

### 1. Limited units and overconfident architecture rankings

**Response.** Agreed. The revised study no longer ranks neural architectures on the small FD1 reader split. It uses 90 complete simulated engine trajectories and five outer engine-held-out folds, with 18 unseen engines in each outer test fold. Four inner engine-grouped folds select the warning threshold using only the corresponding outer-training engines. The claim is limited to this simulated, downsampled dataset and is not presented as an architecture ranking or field-validation result.

### 2. CNN-GRU instability may reflect optimization rather than the architecture family

**Response.** Agreed. Because the original records could not establish the cause of the CNN-GRU variation, we removed the CNN-GRU result and every family-level inference based on it. The revised paper does not use that observation as evidence.

### 3. Selected best-seed result should not drive the comparison

**Response.** Agreed. The selected seed-47 result and its comparison with published work have been removed. The revised paper reports pooled outer-test predictions, complete-engine bootstrap intervals, and five repetitions of the full nested procedure with seeds 2024–2028.

### 4. State limitations directly

**Response.** Implemented. The Discussion explains that tiny-N-CMAPSS is simulated and 100-times downsampled; the 20-cycle boundary and 5% false-alert budget are experimental choices; repeated seeds do not constitute external validation; and the study does not establish a maintenance schedule, fielded-aircraft decision rule, or safety certification.

## Reviewer 2

### 1. The manuscript listed 22 models but reported only three

**Response.** Agreed. The 22-model registry and the claim of a broad architecture benchmark have been removed. The primary comparison now holds one random-forest classifier fixed while changing only the information supplied to it: elapsed cycle, cycle plus operating conditions, measured telemetry only, and all observable inputs. A secondary analysis selects among four prespecified tabular classifiers entirely within each outer-training split.

### 2. The contribution required a narrower and better-supported framing

**Response.** Implemented through a complete narrowing of the research question. The principal result is that, at a matched pre-window alert budget, adding 14 documented measured channels raises final-window recall from 55.4% for cycle plus operating conditions to 79.3% for all observable inputs. The paired complete-engine bootstrap interval for the 23.9-percentage-point recall gain is 18.3 to 29.6 points. This supports an information-value claim on the tested simulator; it does not support a universal model-ranking claim.

### 3. Reproducibility requires an explicit code and data trail

**Response.** Implemented. The manuscript links the project repository and the exact tiny-N-CMAPSS source commit. The evaluation records the source-file hashes, input definitions, outer-test predictions, selected thresholds, model configurations, tables, and figures.

## Reviewer 3

### 1. Reduce the number of headline values in the abstract

**Response.** Implemented. The abstract centers the matched-budget recall result and its complete-engine interval. Supporting AUROC, precision, fixed-threshold, timing, and seed-level results are reported in the Results section rather than packed into the abstract.

### 2. Runtime comparisons require hardware normalization

**Response.** Agreed. The runtime figure and all runtime-based claims have been removed. Runtime is not part of the revised study's evidence.

### 3. Strengthen reproducibility and related-work positioning

**Response.** Implemented. The revised manuscript gives the public code and data links, identifies the exact data revision, and explains why prior RUL-regression scores are context rather than directly comparable baselines for the present binary warning policy.

## Main changes in the revision

- Replaced the architecture-ranking question with a maintenance-warning information-value question.
- Evaluated 90 complete trajectories with nested engine-held-out validation.
- Added strong elapsed-cycle and cycle-plus-operating-condition baselines.
- Corrected the input definition to the 14 documented measured physical channels.
- Excluded auxiliary flight-class and health-state fields, engine identity, and RUL from every model.
- Selected thresholds inside each outer-training split under a prespecified 5% pre-window false-alert budget.
- Added complete-engine bootstrap intervals and five-seed robustness checks.
- Added an independently nested model-family selection analysis.
- Removed the 22-model registry, CNN-GRU reliability inference, selected best seed, unnormalized runtime figure, and direct literature-score comparison.
- Added explicit code, data-version, and reproducibility information.
