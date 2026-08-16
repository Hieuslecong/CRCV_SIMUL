# CRCV V5.4 — Final Audit Report

## Decision

**PROPOSAL QUALIFIED / FINAL RECOVERY FAIL-CLOSED.**

V5.4 resolves the proposal-coverage problem identified in V5.3, but it does **not** solve final path acceptance. Runtime therefore remains Base-only. No final-test result is produced.

## 1. What was fixed

- Replaced contaminated nominal missing-skeleton supervision with **CoreGap** semantics: `GT skeleton AND NOT Dilate(Base, 2px)`. Exact AddedPrecision is unchanged.
- Added component-balanced endpoint selection and border filtering.
- Added **RGB ridge open-CONTINUE** proposals for long tails where learned XY rollout drifts.
- Added top-K spatial target hypotheses to V5.2b instead of retaining only one target per distance band.
- Kept V5.3 iterative tracer as a complementary proposal family.
- Rejected both CONNECT implementations after they failed CAL contribution tests.

## 2. Proposal qualification

On CAL under CoreGap semantics:

- Oracle AddedPrecision: **98.21%**
- CoreGap recovery: **32.86%**
- Core hit: **23/70**
- Recoverable-image coverage: **4/4 = 100.0%**
- Exact TP/FP additions: **55/1**

This satisfies the proposal gate (>=95% precision and >=30% recovery).

For external development context, the earlier real_val oracle under the older regular missing-skeleton denominator reached **98.88% precision / 37.45% recovery / 8/10 image coverage**. It is not used as a calibration metric and is not directly numerically compared to CoreGap recovery.

## 3. Acceptance verifier remains the blocker

The final pairwise same-source verifier was fitted on Module-TRAIN and assessed on CAL.

- Module bank: 24106 candidates / 425 positive.
- CAL bank: 23337 candidates / 63 positive.
- CAL candidate AUC: **0.628**
- CAL AP: **0.0052**

Despite proposal quality, the verifier cannot reliably distinguish the few true paths from hard false paths generated from the same source. No CAL operating point reaches the required >=85% AddedPrecision with zero normal additions. The best explicitly zero-normal point has **0 TP / 3 FP**.

Therefore:

`proposal_qualified = true`

`verifier_qualified = false`

`recovery_enabled = false`

## 4. Active and rejected proposal families

### Retained
- V5.2b multi-target field/geodesic
- V5.2b top-K spatial targets + NMS
- V5.3 iterative tracer
- V5.4 RGB ridge CONTINUE walker

### Rejected
- V5.4 destination-conditioned Dijkstra CONNECT
- V5.4 bidirectional ridge meet CONNECT

The rejected CONNECT variants are not allowed to influence runtime output.

## 5. What must not be enabled yet

- final recovery output
- suppression integration
- width reconstruction
- joint Base+CRCV training

All remain fail-closed until acceptance is qualified.

## 6. Next technical target

Do **not** continue scalar threshold/grid tuning of the handcrafted verifier. The next justified experiment is a **path-aware image relation verifier** trained on the now-frozen V5.4 proposal bank, with same-source hard negatives and direct RGB/path/source/destination evidence. Proposal generation should remain frozen so acceptance contribution can be measured independently.

## 7. Regression / protocol

- 24/24 tests PASS.
- `compileall` PASS.
- final test: **SEALED_NOT_USED**.
- runtime: **FAIL_CLOSED_BASE_ONLY**.
