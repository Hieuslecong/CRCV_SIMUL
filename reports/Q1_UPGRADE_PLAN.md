# CRCV V5.19-Q1 validation upgrade

This release freezes the V5.18.1 method family and changes the research emphasis from architecture iteration to qualification.

## Frozen method
- Full crack proposal / topology recovery.
- RGB-conditioned width-aware ribbon reconstruction.
- Local residual suppression inside Frozen-Base support.
- Metric-aligned action scoring.
- Runtime uses no GT.

## Mandatory qualification matrix
1. Native 128 and 256 training/evaluation. Do not resize a 160 proposal bank.
2. Five backbones, including at least one official reference implementation.
3. Three complete end-to-end training seeds; action-scorer-only seeds are insufficient.
4. At least three datasets; at least one must be independent/external to development.
5. At least two cross-dataset train->test routes.
6. Leave-one-backbone-out qualification.
7. Base + morphology + a published refinement comparator + CRCV.
8. Paired image-level bootstrap CI and paired permutation tests with Holm correction.
9. CPU and edge latency, plus parameter/memory overhead.
10. One-shot final test only after architecture/config/checkpoint/threshold freeze.

## Publication gates
- mean ΔF1 >= +1.0 pp;
- mean ΔmIoU >= +0.5 pp;
- >=80% positive backbone x dataset pairs;
- no pair ΔF1 < -0.2 pp;
- 95% paired bootstrap CI lower bound > 0 for F1 and mIoU;
- Holm-corrected p < 0.05 for F1 and mIoU;
- LOBO and cross-dataset evidence positive;
- final test untouched during development.

The gate checker returns BLOCKED until every evidence field is present. This intentionally prevents development-smoke numbers from being mislabeled as publication evidence.
