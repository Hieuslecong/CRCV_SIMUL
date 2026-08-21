# CRCV V5.20.1 root-cause fix report — 2026-08-21

Branch: `feature/crcv-v5-20-counterfactual-policy`

## Decision

**P0 root causes fixed. Engineering/safety smoke PASS. Real-data policy smoke improved from 0/5 ACTIVE to 4/5 ACTIVE at seed 1337 / 128. Full multi-seed/256 qualification is still blocked until the fix is reproduced from a canonical end-to-end runner and validated at the next ladder stage.**

## Fixed root causes

1. **Counterfactual supervision leakage fixed.** `side_spur` and `false_bridge` now dilate only the newly generated primitive, never the complete GT crack. This prevents synthetic REMOVE labels from forming an artificial halo along the true crack boundary.
2. **Synthetic/Base mismatch reduced.** Counterfactual over-segmentation is intended to be overlaid on the real frozen Base rather than replacing the Base with GT-derived foreground.
3. **REMOVE over-weighting removed.** Canonical V5.20.1 training uses KEEP-heavy sampling and does not use `class_weight='balanced'`.
4. **Hard KEEP supervision added.** Boundary, endpoint and junction neighborhoods are oversampled as KEEP; a one-pixel GT tolerance band is excluded from REMOVE labels.
5. **Topology/scale-aware features added.** Runtime features now include normalized skeleton distance, local radial position, endpoint/junction distance, nearest skeleton degree, component area, skeleton length and elongation. Absolute x/y position was removed.
6. **FIT-only error profiling added.** Natural FP/FN component statistics can calibrate counterfactual scale parameters rather than relying only on fixed smoke defaults.
7. **Component-level fallback added.** Whole small Base components can be removed atomically; partial crack-edge shaving is impossible in this family.
8. **CAL/VAL safety aligned.** Qualification requires non-regressive Dice, Recall loss >= -0.002, TCRR <= 0.003 and FPRR >= 0.005.
9. **Safety-first operating point committed.** Pixel thresholds prefer the highest CAL-qualified threshold; pixel/component family selection is CAL-only. VAL remains fail-closed qualification.
10. **Canonical policy-training recipe committed.** Learner hyperparameters, sampling ratios and counterfactual probability-lift ranges are now code rather than undocumented smoke state.

## Regression smoke

Local updated modules passed **14/14** V5.20.1 unit/regression tests before repository synchronization. Tests cover primitive-only corruption, topology-aware feature contract, boundary-tolerance REMOVE labels, KEEP-heavy sampling, component-level removal, error-profile calibration, long-boundary protection, GT-free runtime, TCRR/FPRR and fail-closed qualification.

## Real-data smoke after fixes

Setup: existing real Base checkpoints, seed 1337, 128x128, FIT=48, CAL=12, VAL=20, DEV_TEST=10. DEV_TEST has been opened previously and is not publication evidence.

A safety-first hybrid was tested: CAL selects between (A) the learned pixel-candidate policy at the most conservative qualified threshold and (B) a whole-component fallback. VAL then independently enables/disables the selected family.

| Backbone | CAL-selected family | VAL status | VAL dDice | VAL dRecall | VAL TCRR | VAL FPRR |
|---|---|---|---:|---:|---:|---:|
| TinyUNet | pixel | ACTIVE | +0.00301 | -0.00127 | 0.00217 | 0.02313 |
| FastSCNNLite | component | ACTIVE | +0.00035 | -0.00073 | 0.00236 | 0.00947 |
| BiSeNetTiny | component | NO_OP | +0.00080 | +0.00000 | 0.00000 | 0.00424 |
| MobileNetV3SmallSeg | component | ACTIVE | +0.00095 | -0.00036 | 0.00063 | 0.00702 |
| DSUNetLite | pixel | ACTIVE | +0.00428 | -0.00073 | 0.00121 | 0.02454 |

Outcome: **4/5 ACTIVE**. BiSeNetTiny remains fail-closed only because useful-removal rate on VAL (FPRR 0.00424) is slightly below the fixed 0.005 utility floor; importantly its VAL TCRR is zero.

Effective deployed development-test mean across five backbones (NO_OP counted as exact zero mutation):

- dDice: **+0.000954** (+0.095 pp)
- dCrack-IoU: **+0.000756**
- dRecall: **-0.000409** (-0.041 pp)
- TCRR: **0.000838** (0.084%)
- FPRR: **0.008853** (0.885%)
- ACTIVE: **4/5**

This is materially safer than the pre-fix independent rerun, where all five policies failed qualification and raw VAL TCRR ranged roughly 0.98% to 2.31%.

## Remaining blockers

- The full executable dataset/Base/policy orchestration still needs to be consolidated into one committed CLI runner; the canonical policy recipe and selector are now committed, but the historical Base definitions/data loading remain separate.
- The FIT error profiler must be wired into training so counterfactual parameters are actually generated per backbone/seed instead of merely being available as a module.
- 4/5 ACTIVE is a smoke milestone, not evidence of model-agnostic generalization.
- Do not expand publication claims from the opened DEV_TEST.
- Next ladder: reproduce seed1337/128 from a clean checkout, then seed1337/256; only then run seeds 2027/31415.
- ADD/recovery remains outside this REMOVE-focused fix and must be integrated later as a separate validated stage.
