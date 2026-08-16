# CRCV V5.4 — Round 4: Acceptance verifier audit

## Data
Module-TRAIN path bank after CoreGap + top-K augmentation:
- candidates: 24106
- positive: 425
- negative: 22760
- ambiguous: 921

CAL bank:
- candidates: 23337
- positive: 63
- negative: 22836
- ambiguous: 438

## Pairwise same-source verifier
CAL candidate diagnostics:
- AUC: **0.628**
- AP: **0.0052**

The class imbalance and hard same-source false paths remain severe. No image-level operating point reaches AddedPrecision >=85% with zero normal additions. The unrestricted fallback point has precision only **12.5%** and adds 50 pixels to CAL normals. An explicit zero-normal scan yields TP=0 / FP=3.

## Decision
**Verifier FAIL.** Stop scalar threshold tuning. A path-aware image relation representation is required if development continues.
