# CRCV V5.7–V5.8 Recovery Acceptance Audit

## Decision

**RECOVERY REMAINS NOT QUALIFIED. DO NOT ADD MORE PROPOSAL FAMILIES OR ENABLE RECOVERY.**

The ordered path-aligned verifier hypothesis was implemented and tested, then repeated with a six-fold OOF upstream construction closer to the final Base training distribution. Neither version reaches the predeclared recovery gate. A covariate-shift audit shows that the verifier-training upstream distribution is materially different from the final frozen Base distribution, and a simple Module-only family-specific structural rule also fails when frozen and transferred to CAL.

The final test remains `SEALED_NOT_USED`.

## 1. V5.7 path-aligned strip verifier

The square source/path/destination crop used in V5.5/V5.6 discards trajectory order. V5.7 therefore samples cross-sections perpendicular to the ordered candidate path and applies a lightweight sequential encoder.

Inputs along the path include real RGB, Base probability/mask, ridge and gradient evidence, candidate role/location, plus a weak simulation geometry score. Simulation remains a geometry prior only.

Implementation:
- `crcv52/path_strip_v57.py`
- ~19.3k parameters
- path order preserved
- same-source ranking + Top-1/margin/abstention

Module-only seed 5701 on the fresh reconstructed bank:
- CAL candidate AUC: **0.5983**
- CAL AP: **0.00905**
- best zero-normal AddedPrecision: **20%**
- CoreGap recovery: **0%**
- TP/FP: **2/8**

**V5.7 FAIL.**

## 2. V5.7b topology-aware strip

V5.7b adds GT-free structural evidence along the ordered path: distance-to-Base, source tangent continuity, maximum/end Base distance, fraction beyond 2 px, distance progression, and fraction close to Base. Model size is **23,356 parameters**.

An aggregate-feature ExtraTrees diagnostic reached approximately AUC **0.816** / AP **0.0334**, but the safest zero-normal top-ranked candidate was still false (TP=0, FP=3). Better candidate-level AUC did not translate into a safe recovery operating point.

## 3. Full natural OOF expansion (V5.7c)

A 3-fold lineage-disjoint OOF Base was constructed over all 36 Base-FIT records, with CAL untouched. Fresh proposal generation produced 31,239 total candidates and 717 positives before filtering. After source>=0.35 and length<=24: **9,718 candidates / 668 positives**, spanning **26 positive source groups / 11 positive images**.

V5.7c seed 5711, 20 epochs:
- training samples: 3,166
- positives: 745
- training images: 41
- CAL AUC: **0.56814**
- CAL AP: **0.00815**
- best zero-normal: TP=0, FP=4, CoreGap recovery=0%

**V5.7c FAIL.**

## 4. Six-fold closer-OOF experiment (V5.8)

A second OOF construction used **6 folds**, so each OOF Base trains on 30/36 Base-FIT records rather than 24/36. This moves the upstream distribution closer to the final frozen Base without using in-sample final-Base errors for verifier training.

The KS statistic for positive-probability mean versus CAL improved from **1.000 (3-fold)** to **0.333 (6-fold)**, but substantial shift remains in overall probability mean (**KS 0.889**) and component count (**KS 0.611**).

Fresh six-fold OOF bank:
- 33,267 total candidates
- 390 positives before eligibility filtering
- **11,006 eligible / 387 positive** after source>=0.35 and length<=24
- **16 positive source groups / 7 positive images**

V5.8 path-strip training (OOF6 + Module), seed 5811, 20 epochs:
- training samples: **2,982**
- positive training samples: **464**
- training images: **45**
- CAL AUC: **0.35504**
- CAL AP: **0.00530**
- best zero-normal AddedPrecision: **11.11%**
- CoreGap recovery: **0%**
- TP/FP: **2/16**

**V5.8 FAIL.** Making OOF more final-like does not rescue the learned acceptance verifier.

## 5. Covariate-shift evidence

Key KS statistics versus CAL:

| Source | positive fraction | prob mean | positive prob mean | component count | Dice |
|---|---:|---:|---:|---:|---:|
| 3-fold OOF | 0.778 | 1.000 | 1.000 | 0.778 | 0.278 |
| 6-fold OOF | 0.667 | 0.889 | 0.333 | 0.611 | 0.333 |
| Module final-Base | 0.333 | 0.333 | 0.500 | 0.667 | 0.500 |

Candidate-family behavior is also unstable across OOF/Module/CAL.

## 6. Simple family-specific gate sanity check

All selection was done on Module only and then frozen on CAL. The only single-scalar rule able to reach >=85% AddedPrecision with zero normal additions on Module was `ridge_continue`, top raw-score candidate per source, threshold **0.8675333**.

Module: AddedPrecision **100%**, CoreGap recovery **9.375%**, TP/FP 6/0.

Frozen unchanged on CAL: AddedPrecision **0%**, CoreGap recovery **0%**, TP/FP 0/6.

Thus a simple family-specific threshold also fails transfer.

## 7. Interpretation

Failure is no longer plausibly explained only by insufficient proposal coverage, loss of path order, too few OOF positives, too-small OOF training fraction, or lack of family-specific calibration. All were tested and none produced safe transfer.

The strongest current conclusion is that **recovery acceptance is not robust under the available frozen-Base development regime**. Proposal oracle capability remains useful diagnostic evidence, not deployed recovery performance.

## 8. Scientific state

```text
V5.4 proposal oracle              = QUALIFIED / FROZEN
V5.5c suppression Original-V52   = CAL PASS, not multi-backbone qualified
simulation -> suppression         = NOT SUPPORTED
simulation -> recovery geometry   = RETAINED
V5.7 ordered strip verifier       = FAIL
V5.7c 3-fold full OOF verifier    = FAIL
V5.8 6-fold closer-OOF verifier   = FAIL
family-specific structural gate  = FAIL transfer
recovery_enabled                  = false
suppression_enabled               = false
runtime                           = FAIL_CLOSED_BASE_ONLY
final_test                        = SEALED_NOT_USED
```

## 9. Next gate

Do **not** stack another recovery model immediately. Freeze V5.4 proposal, keep recovery OFF, obtain a substantially larger matched final-Base-distribution natural error bank or redesign the protocol so verifier-training and deployment Base distributions match without leakage. In parallel, qualify the GT-free suppression baseline on properly trained frozen backbones using a fresh validation set. Only reopen recovery learning when enough matched natural positive source events exist.
