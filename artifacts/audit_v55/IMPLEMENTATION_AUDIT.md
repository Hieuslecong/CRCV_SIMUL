# CRCV V5.5 Implementation Audit

## Decision

**CODE GATE PASS / SCIENTIFIC QUALIFICATION NOT RUN**

This branch implements the next justified CRCV block while preserving V5.4 as the frozen baseline.

## Simulation input audit

The supplied XY file was found and parsed with the existing CRCV separator semantics: literal `0,0` terminates one trajectory. No blank-line grouping is treated as a simulation family.

Profile actually fitted during implementation:

- valid trajectories: 761
- resampled trajectories: 761
- turning samples: 20849
- turn median: 0.049667 rad
- turn 95th percentile: 0.632513 rad
- trajectory tortuosity median: 1.045319
- trajectory tortuosity 95th percentile: 1.329819

Important correction: raw trajectories contain repeated/nearly repeated coordinates, so the V5.5 prior uses per-trajectory arc-length resampling before calculating turn statistics. Simulation remains a geometry prior only; it provides no RGB or width evidence.

## Implemented modules

- `crcv52/sim_prior.py`
  - robust resampling
  - fitted simulation profile
  - bounded path plausibility score
- `crcv52/relational_v55.py`
  - source/path/destination RGB relation encoder
  - same-source hard-negative ranking loss
  - Top-1 + margin + abstention
  - simulation-aware component suppression head
- `crcv52/v55_data.py`
  - reads the frozen V5.4 Module-TRAIN/CAL candidate banks
  - reconstructs ordered candidate paths from packed masks without GT inputs
  - conservative component labels for suppression
- `crcv52/runtime_v55.py`
  - independent qualification gates
  - fail-closed runtime
  - isolated-add and CC-increase structural safety checks
- training/calibration scripts for recovery and suppression

## Complexity

V5.5 relational block parameters: **64,684**, below the predeclared 250k budget.

## Local code tests

New V5.5 test suite: **9/9 PASS**.

Covered:

1. simulation prior ranks smooth crack-like geometry above zigzag corruption,
2. relation input schema,
3. parameter budget and output heads,
4. same-source ranking behavior,
5. margin abstention,
6. component-view schema,
7. exact fail-closed runtime,
8. rejection of unqualified enabled recovery,
9. isolated-add safety rejection.

`py_compile` also passed for all new V5.5 modules/scripts. The training script was additionally corrected to keep each same-source group and its hard negatives inside the same mini-batch, preventing the ranking term from being accidentally zeroed by random batch splitting.

## Scientific state

The GitHub release does not contain the certified V5.4 OOF cache, so the learned V5.5 relation/suppression heads have **not** been trained or CAL-qualified in this branch yet.

Therefore:

```text
proposal_qualified          = true   # inherited/frozen V5.4
sim_prior_profile_fitted    = true
relation_verifier_qualified = false
suppression_qualified       = false
recovery_enabled            = false
suppression_enabled         = false
final_test                  = SEALED_NOT_USED
```

The next run must take place in the certified workspace containing:
`artifacts/cache/v52_module.pkl`, `v52_cal.pkl`, and `v54_*_coregap_topk_parts/`.
