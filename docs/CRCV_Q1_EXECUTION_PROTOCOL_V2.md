# CRCV Q1 Publication Execution Protocol V2

## Purpose

This document freezes the execution contract for turning CRCV V5.18.1 from development evidence into a publication-grade evidence package. The goal is scientific validity and reproducibility, not a software label that predicts journal acceptance.

## Phase 0 — Historical-data and provenance freeze

### Inputs
- every historical train/CAL/VAL/test manifest used by V5.x;
- all upstream proposal/ribbon/authenticity/action checkpoints;
- all result artifacts used to make architecture or threshold decisions.

### Required actions
1. Build one canonical sample manifest with:
   `sample_id, source_dataset, lineage_id, parent_id, acquisition_id, split, historically_exposed`.
2. Mark every sample ever inspected or used in V5.x as `historically_exposed=true`.
3. Treat all historically exposed data as DEVELOPMENT data.
4. Build a checkpoint provenance ledger with training manifest SHA256, split manifest SHA256, config SHA256, code commit and checkpoint SHA256.
5. Any checkpoint with unknown future-test provenance must be retrained or excluded from final evidence.

### Pass condition
- no final/external sample is historically exposed;
- no final lineage occurs in development;
- every retained upstream checkpoint has complete provenance.

## Phase 1 — Method freeze and scale normalization

### Frozen scientific core
- ADD/recovery branch;
- RGB/width-aware ribbon reconstruction;
- REMOVE/local residual suppression;
- shared action scoring over frozen Base predictions.

### Required corrections before final freeze
- remove backbone-specific `WEAK` weighting from the publication model;
- express geometry/radius/action budgets in scale-aware or crack-width-aware units where possible;
- keep conservative suppression bounded by both image area and predicted-foreground area;
- separate training-only GT-derived candidate targets from runtime candidate fields.

### Pass condition
- no architecture-specific exception is needed to obtain the claimed generalization;
- runtime representation contains no GT-derived target fields;
- deterministic smoke tests pass at 128 and 256.

## Phase 2 — Strong Base qualification

Use at least five segmentation backbones, including at least two canonical/reference implementations. Train each Base model to convergence or provide a defensible early-stopping criterion.

Required full seeds: `1337`, `2027`, `31415`.

Store for every Base run:
- git commit;
- dataset/split/config hashes;
- seed;
- checkpoint hash;
- threshold-selection artifact;
- per-image validation metrics;
- training curve/convergence artifact.

## Phase 3 — In-domain CRCV qualification

For each qualified Base prediction set, evaluate under exactly matched data, seed and threshold-selection policy:
- Base;
- Base + Morphology;
- published refiner (StripCuts or a directly equivalent crack-refinement method);
- CRCV V5.18.1.

Primary endpoints:
- Dice/F1;
- crack-foreground IoU.

Secondary endpoints:
- Precision;
- Recall;
- 2-class mIoU;
- structural/topology metric;
- background false-positive rate.

## Phase 4 — Ablation

Minimum ablation:
- Base;
- Base + REMOVE;
- Base + ADD centerline;
- Base + ADD width prior;
- Base + ADD RGB ribbon;
- Base + ADD + REMOVE.

All variants must reuse the same frozen Base predictions and the same split/calibration policy.

## Phase 5 — LOBO unseen-backbone qualification

For every declared backbone B_i:
1. train the CRCV learned correction components without B_i;
2. freeze them;
3. evaluate on B_i predictions without target-backbone retraining or target-backbone-specific weighting.

Report each held-out backbone separately. Do not hide a failed architecture behind the mean.

## Phase 6 — Cross-dataset qualification

Run at least two source-disjoint routes. A valid route has:
- training source families that do not overlap the target source family;
- no target-domain retraining;
- no target-domain threshold calibration when making a zero-shot claim.

Pooling several datasets and performing another mixed split is not cross-dataset evidence.

## Phase 7 — Runtime GT-independence proof

Run an end-to-end mutation test:
1. inference with a normal runtime record;
2. inference after deleting/randomizing all GT and all GT-derived target fields;
3. assert bit-identical or numerically equivalent final predictions.

The test must cover proposal, ribbon, authenticity, action scoring, ADD, REMOVE and final mask production.

## Phase 8 — Statistical analysis

Use lineage/parent as the resampling unit. Individual crops, repeated seeds or multiple backbone outputs from the same parent are not independent observations.

Primary hypotheses:
- H1: paired cluster-level Dice improvement > 0;
- H2: paired cluster-level crack-IoU improvement > 0.

Required:
- paired cluster bootstrap 95% CI;
- paired cluster sign-permutation test;
- Holm correction across the two primary hypotheses;
- mean, median, CI, worst pair and positive dataset×backbone rate.

## Phase 9 — Efficiency

Always report CPU latency. Report edge latency only if making an edge/lightweight deployment claim.

For each device:
- warm-up;
- at least 100 timed runs;
- median, mean, standard deviation and P95;
- Base latency;
- CRCV overhead;
- total latency;
- peak memory where available.

## Phase 10 — Fresh external one-shot test

The final external test must be:
- fresh;
- source/lineage-disjoint;
- `historically_exposed=false`;
- unopened until architecture, checkpoints, thresholds and protocol are frozen.

After one-shot evaluation, no architecture or threshold change may be justified using that test. Any post-test method change creates a new method version and requires a new fresh confirmation source.

## Evidence status

`EVIDENCE_COMPLETE` means only that the frozen protocol requirements are satisfied and all required artifacts are present and hash-valid. It must never be described as automatic Q1 acceptance readiness.

## Stop conditions

Stop or substantially redesign CRCV if, after validity repair:
- gains disappear on converged strong Base models;
- LOBO is consistently negative;
- at least two independent cross-domain routes are negative;
- a direct published refiner dominates CRCV in accuracy/topology/cost;
- fresh external effect is approximately zero or negative;
- false-bridge/topology damage increases materially despite pixel-metric gains;
- required upstream checkpoint provenance cannot be reconstructed or retrained.
