# CRCV V5.18.1 — Q1 Multi-Dataset Assessment and Qualification Plan

## 1. Executive assessment

CRCV V5.18.1 currently shows strong **development-level potential** for a Q1 journal paper, but it is **not yet publication-ready**. The clean retrain has reproduced the previous V5.18.1 development gains and confirms that the method improves multiple lightweight backbones under the current controlled protocol.

Current clean-retrain aggregate over three action-scorer seeds:

- ΔPrecision: **+4.367 ± 0.145 pp**
- ΔRecall: **+4.819 ± 0.088 pp**
- ΔF1: **+4.611 ± 0.045 pp**
- ΔmIoU: **+1.827 ± 0.020 pp**
- 5/5 development backbones show positive Recall, F1 and mIoU across all three repeated scorer seeds.
- Final/test remains sealed.

The correct next step is **not another architecture revision**. The priority is to prove that the frozen V5.18.1 correction mechanism generalizes across datasets, backbones, resolutions and unseen domains.

---

## 2. Current method positioning

The intended publication positioning should be more specific than “connect broken cracks and remove noise”. A stronger formulation is:

> **CRCV is a corrective segmentation refinement framework that explicitly performs learned ADD and REMOVE operations on frozen crack predictions, using RGB-constrained width-aware reconstruction for missing structures and risk-aware local residual suppression for false foreground.**

The two corrective branches have distinct roles:

- **ADD / Recovery**: recover missed crack regions using directional/path proposals, RGB evidence, destination/source verification and width-aware ribbon reconstruction.
- **REMOVE / Suppression**: remove local false-foreground residuals conservatively while preserving true crack pixels.

Runtime must remain GT-free. GT is allowed only for offline training targets, labels, diagnostics and oracle analysis.

---

## 3. Why training on more datasets can make this Q1-level

Simply increasing the number of training images is not sufficient. The multi-dataset study must demonstrate:

1. **source/lineage-disjoint data integrity**;
2. **consistent improvement across datasets**;
3. **cross-dataset generalization**;
4. **multi-backbone robustness**;
5. **full-seed reproducibility**;
6. **paired statistical significance**;
7. **fair comparison with simple and published refinement baselines**;
8. **runtime and edge-device cost**;
9. **transparent failure analysis**.

A smaller development-set gain of +4.6 pp F1 is useful evidence, but a stable +1–3 pp gain on large, independent, multi-dataset experiments would be substantially more convincing for publication.

---

## 4. Recommended public dataset package

Recommended main benchmark pool:

- **Crack500**
- **DeepCrack**
- **CFD**
- **OmniCrack30k**

A controlled OmniCrack30k subset may be used first for development, followed by larger-scale experiments.

### Critical split rule

Do **not** perform image-random train/test splitting when source lineage is available.

Use the strongest available grouping unit:

```text
source / parent / acquisition session / lineage
```

instead of:

```text
random image
```

If one benchmark is included as a source within a larger aggregated dataset, that source must not also be reported as an unseen external test dataset after training on it.

---

## 5. Proposed Q1 experiment matrix

### 5.1 In-domain multi-dataset experiment

For each public dataset:

```text
Dataset
  × Backbone
  × Seed
  × Resolution
  × Method
```

Recommended minimum:

- datasets: **≥3**, preferably 4;
- backbones: **≥5**;
- full training seeds: **1337, 2027, 31415**;
- main resolution: **256×256**;
- resolution ablation: **128×128 and 512×512** where feasible.

### 5.2 Cross-dataset evaluation

Example route:

```text
TRAIN:
Crack500 + DeepCrack + CFD

TEST:
source-disjoint OmniCrack30k subset
```

Additional material/domain route:

```text
TRAIN:
concrete + asphalt + masonry sources

TEST:
unseen source/material/domain
```

This experiment is more important than simply pooling all datasets and performing another random split.

### 5.3 Leave-one-backbone-out (LOBO)

If the paper wants to claim model-agnostic or plug-in behavior:

```text
train CRCV action model on 4 backbones
→ evaluate on the 5th unseen backbone
```

Every held-out backbone should be reported separately. A positive average does not hide a catastrophic failure on one architecture.

Until this gate passes, use wording such as **shared multi-backbone corrective refinement**, not strong universal/backbone-agnostic claims.

---

## 6. Required comparator package

At minimum compare:

1. Frozen Base segmentation;
2. morphological closing/opening;
3. simple endpoint/bridge baseline;
4. shortest-path or geodesic connector where applicable;
5. published segmentation-refinement comparator;
6. CRCV recovery-only;
7. CRCV suppression-only;
8. full CRCV V5.18.1.

A particularly important direct comparator is a recent crack-specific refinement method such as **Minimum Strip Cuts** if reproducible under the same Base predictions and datasets.

Do not remove a baseline because it performs better than CRCV on a metric. All relevant outcomes must be reported.

---

## 7. Required ablation structure

Recommended ablations:

| Variant | Purpose |
|---|---|
| Base | Frozen segmentation baseline |
| Base + Morphology | Simple non-learned reference |
| Base + REMOVE | Suppression contribution |
| Base + ADD centerline | Recovery without width reconstruction |
| Base + source-width ribbon | Geometry width prior |
| Base + learned RGB ribbon | Width-aware learned recovery |
| Base + ADD + REMOVE | Full CRCV |

Additional diagnostics may include topology/width metrics, but the primary paper metrics remain:

- Precision
- Recall
- F1 / Dice
- mIoU

---

## 8. Publication qualification gates

Suggested practical gates for the large independent benchmark:

```text
mean ΔF1   >= +1.0 pp
mean ΔmIoU >= +0.5 pp
```

and:

```text
>= 80% of dataset × backbone pairs positive
```

with no catastrophic Recall collapse.

For paired image-level statistics:

```text
95% CI lower bound of ΔF1   > 0
95% CI lower bound of ΔmIoU > 0
```

Use paired bootstrap and/or paired permutation/Wilcoxon tests. Apply multiple-comparison correction when evaluating many dataset/backbone combinations.

These values are **project qualification targets**, not formal journal acceptance thresholds.

---

## 9. Full-seed requirement

The current three repeated seeds mainly verify action-scorer stability. Publication-grade robustness should include **full Base → CRCV retraining** for each seed.

Required seeds:

```text
1337
2027
31415
```

Each seed should independently reproduce:

```text
raw data
→ split
→ Base training
→ candidate/action bank
→ RGB ribbon model
→ foreground-authenticity model
→ action scorer
→ CAL threshold selection
→ VAL/external evaluation
```

Report mean ± standard deviation across full runs.

---

## 10. Efficiency and edge-device evidence

For each deployment resolution report:

```text
Base latency
CRCV overhead
Total latency
Peak memory
```

Recommended devices:

- CPU reference;
- desktop/server GPU where available;
- **Jetson Orin Nano** if the paper claims lightweight edge deployment.

Use warm-up and at least approximately 100 timed inference runs per condition.

The proposal-generation stage should be profiled separately because it has historically dominated CPU runtime.

---

## 11. Failure analysis required for a strong paper

Do not show only favorable examples.

A publication qualitative panel should contain at least:

```text
5 best cases
5 recovery-dominant cases
5 background/noise cases
5 worst/failure cases
```

Recommended columns:

```text
RGB | GT | Base | CRCV | Added | Removed | Error Map
```

Current qualitative evidence suggests that V5.18.1 is effective on local false positives and missing crack recovery, but highly textured false-foreground regions remain an important failure mode. This should be reported rather than hidden.

---

## 12. Recommended journal-readiness scenarios

### Scenario A — Only more training data

```text
1 dataset → several datasets
random split
no cross-dataset
no strong comparator
```

**Assessment: insufficient for a strong Q1 claim.**

### Scenario B — Proper multi-dataset qualification

```text
3–4 datasets
3 full seeds
5 backbones
source-disjoint splits
cross-dataset evaluation
paired statistics
```

**Assessment: realistic Q1 submission potential.**

### Scenario C — Full qualification package

```text
4 datasets
3 full seeds
5 backbones
cross-dataset
LOBO
published refinement comparator
complete ablations
failure analysis
CPU/GPU/Jetson profiling
final one-shot test
```

**Assessment: strong Q1-ready evidence package if the gains remain stable.**

---

## 13. Recommended next execution order

Freeze the V5.18.1 architecture and execute:

```text
P0  Self-contained 160×160 reproduction
 ↓
P1  3 full Base→CRCV seeds
 ↓
P2  Native 128 / 256 / 512 resolution study
 ↓
P3  3–4 public datasets with lineage-safe splits
 ↓
P4  Cross-dataset evaluation
 ↓
P5  LOBO unseen-backbone evaluation
 ↓
P6  Morphology + published refinement comparators
 ↓
P7  Paired statistical analysis
 ↓
P8  CPU / GPU / Jetson latency and memory
 ↓
P9  Freeze code/config/checkpoint/manifest hashes
 ↓
P10 Final test one-shot
```

Do not introduce another architecture version unless one of these qualification experiments identifies a concrete failure that cannot be addressed through protocol, data coverage or calibration.

---

## 14. Current verdict

```text
METHOD POTENTIAL        = STRONG
CURRENT Q1 EVIDENCE     = INCOMPLETE
ARCHITECTURE STATUS     = FREEZE V5.18.1
NEXT PRIORITY           = MULTI-DATASET GENERALIZATION
FINAL TEST              = SEALED
```

If V5.18.1 maintains approximately **+1–3 pp F1**, a positive mIoU gain, statistically positive paired confidence intervals, and broad positive coverage across independent datasets/backbones, the project has a credible foundation for a serious Q1 submission.
