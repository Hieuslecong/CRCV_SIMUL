# CRCV V5.20.2 promoted-fix smoke report — 2026-08-21

Branch: `feature/crcv-v5-20-counterfactual-policy`

## Decision

**Engineering/safety smoke: PASS. Scientific effect: SMALL. Continue before multi-seed publication experiments.**

Promoted changes:

- scale-free whole-component fallback: removed absolute `max_pixels`;
- FPRR separated from safety qualification (effect size, not universal safety floor);
- exact no-op is not labelled ACTIVE when removal counts are available;
- operating-point selection changed from arbitrary weighted utility to constrained lexicographic selection;
- FIT error profiler extended with scale-normalized area, skeleton length and distance-to-reference-width statistics;
- V5.20.1 pixel policy remains the stable fallback; unsafe experimental region learners are not promoted.

## Unit/regression smoke

- `pytest tests/test_v520_counterfactual.py`: **17/17 PASS** in the local mirror of the promoted changes.
- `compileall`: PASS.
- Added explicit 128/256 scale-equivalence regression for component fallback.

## Real-data smoke: seed 1337 / 128

After removing the arbitrary default FPRR floor, all five policies satisfy the safety gate on VAL. This does **not** mean all five have meaningful effect size; DSUNetLite is practically near-no-op on the development TEST.

- ACTIVE: **5/5**
- deployed TEST mean ΔDice: **0.001303** (0.130 pp)
- deployed TEST mean ΔCrack-IoU: **0.001022**
- deployed TEST mean ΔRecall: **-0.000409**
- deployed TEST mean TCRR: **0.000838** (0.084%)
- deployed TEST mean FPRR: **0.010505**

## Real-data smoke: seed 1337 / 256

Scale-free calibration was evaluated independently per backbone using CAL-only operating-point selection and VAL fail-closed qualification.

- ACTIVE: **4/5**
- TinyUNet: NO-OP (VAL Dice regression at otherwise safe low-mutation point)
- FastSCNNLite: ACTIVE, scale-free component fallback
- BiSeNetTiny: ACTIVE, pixel policy
- MobileNetV3SmallSeg: ACTIVE, pixel policy
- DSUNetLite: ACTIVE, pixel policy
- deployed TEST mean ΔDice: **0.000490** (0.049 pp)
- deployed TEST mean ΔCrack-IoU: **0.000426**
- deployed TEST mean ΔRecall: **-0.000164**
- deployed TEST mean TCRR: **0.000283** (0.028%)
- deployed TEST mean FPRR: **0.002945**

## Experimental paths rejected, not promoted

1. **Forcing known attached synthetic FP directly into pixel REMOVE labels**: degraded the 128 smoke to roughly **1/5 ACTIVE**. This confirms that attached-FP supervision needs region semantics rather than naive pixel relabelling.
2. **Learned candidate-region classifier**: early smoke produced excessive TCRR / NO-OP behavior. Not promoted.
3. **Learned whole-component classifier**: sparse/imbalanced component labels led to 0–3/5 ACTIVE and near-no-op or negative effective behavior. Not promoted.

These negative results are important: V5.20.2 keeps the safe V5.20.1 pixel policy plus scale-free atomic-component fallback rather than replacing a stable path with an unvalidated learned region model.

## Remaining blockers

- effect size remains small, especially at 256;
- TinyUNet@256 remains fail-closed NO-OP;
- region-level attached-spur/over-width/false-bridge semantics are not yet solved by a robust learned policy;
- ADD is not integrated;
- only seed 1337 has been evaluated after these fixes;
- no LOBO/cross-dataset/fresh external final evidence yet.

The opened `real_debug_data/test` remains development evidence only, not a publication holdout.
