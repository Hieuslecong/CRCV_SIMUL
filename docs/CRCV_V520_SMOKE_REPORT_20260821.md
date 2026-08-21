# CRCV V5.20 counterfactual corrective policy — smoke report (2026-08-21)

Branch: `feature/crcv-v5-20-counterfactual-policy`

## Decision

**Engineering design smoke: PASS. Real-data REMOVE-policy smoke: PARTIAL PASS / CONTINUE REDESIGN.**

V5.20 replaces foreground-authenticity-as-removal with explicit corrective supervision:

- `KEEP = Base ∩ GT`
- `REMOVE = Base \ GT`
- `ADD = GT \ Base`

Synthetic/simulated cracks are now used to generate controlled counterfactual Base errors rather than merely providing crack-like appearance samples.

## Implemented modules

- `crcv52/action_targets.py`: exact ADD/KEEP/REMOVE/IGNORE targets.
- `crcv52/counterfactual_errors.py`: auditable corruption operators (`gap_delete`, `endpoint_truncate`, `boundary_erode`, `width_dilate`, `side_spur`, `isolated_blob`, `false_bridge`).
- `crcv52/removal_policy.py`: GT-free appearance + Base-structural feature contract for KEEP/REMOVE learning.
- `crcv52/safety_projection.py`: GT-free core protection, deletion budgets, connected-component preservation and parallel-boundary-strip rejection.
- `crcv_q1/action_metrics.py`: Dice/Crack-IoU plus TCRR/FPRR safety metrics.
- `crcv_q1/remove_qualification.py`: VAL-only fail-closed activation gate.
- `tests/test_v520_counterfactual.py`: action-label, corruption, feature, runtime-GT-independence, boundary-strip, TCRR/FPRR and fail-closed tests.

## Real-data smoke setup

This smoke reused the already trained **seed 1337, 128×128** Base checkpoints from the real-data CPU run:

- TinyUNet
- FastSCNNLite
- BiSeNetTiny
- MobileNetV3SmallSeg
- DSUNetLite

Data roles:

- real train: deterministic FIT=48 / CAL=12
- VAL=20
- TEST=10

The TEST split was previously opened by explicit request, so all results here are **development evidence**, not final publication evidence.

The V5.20 KEEP/REMOVE learner was trained from:

1. natural Base errors on FIT;
2. counterfactual over-segmentation generated from FIT GT masks (dilation/spur/blob families).

Base threshold was selected on CAL. REMOVE threshold was selected on CAL under safety constraints. No TEST data were used for threshold selection.

## Smoke 1 — policy + core safety projection

Before the explicit parallel-boundary guard, mean across five backbones on TEST was approximately:

- ΔF1: **+0.0099** (+0.99 pp)
- ΔCrack-IoU: **+0.0079**
- ΔRecall: **−0.0042**
- TCRR: **0.0074** (0.74% of already-correct Base crack removed)
- FPRR: **0.0519**

This was not accepted as a safety pass because TCRR was too large.

## Smoke 2 — parallel boundary-strip protection

`SafetyProjection` was upgraded to reject long high-REMOVE-score regions whose own skeleton is long and whose median distance to the Base skeleton is small. This directly targets the observed failure mode where REMOVE shaves a long strip from a true crack edge.

Mean TEST result across five backbones after the guard:

- ΔF1: **+0.0088** (+0.88 pp)
- ΔCrack-IoU: **+0.0069**
- ΔRecall: **−0.0038**
- TCRR: **0.0060** (0.60%)
- FPRR: **0.0463**

TCRR improved by roughly 19%, but the result is still not sufficiently safe as an always-on policy.

## Smoke 3 — VAL-only fail-closed qualification

A second qualification stage was added after CAL selection. A REMOVE policy is enabled only when VAL satisfies all of:

- ΔDice ≥ −0.001
- ΔRecall ≥ −0.002
- TCRR ≤ 0.003
- FPRR ≥ 0.005

Outcome at seed 1337 / 128:

| Backbone | VAL ΔF1 | VAL ΔRecall | VAL TCRR | VAL FPRR | Runtime status |
|---|---:|---:|---:|---:|---|
| TinyUNet | +0.00004 | +0.00000 | 0.00000 | 0.00008 | NO-OP (insufficient useful removal) |
| FastSCNNLite | +0.00312 | −0.00798 | 0.01071 | 0.03965 | NO-OP |
| BiSeNetTiny | +0.01681 | −0.00690 | 0.00566 | 0.03797 | NO-OP |
| MobileNetV3SmallSeg | +0.01258 | −0.00563 | 0.00437 | 0.03580 | NO-OP |
| DSUNetLite | +0.00195 | −0.00018 | 0.00006 | 0.00807 | **ACTIVE** |

Therefore **1/5 policies are active and 4/5 fail closed to the unchanged Base mask**.

For the one qualified DSUNetLite policy, development TEST behavior was approximately:

- ΔF1: +0.0050
- ΔCrack-IoU: +0.0030
- ΔRecall: −0.0003
- TCRR: 0.00054
- FPRR: 0.0117

If NO-OP backbones are included as exact zero-delta deployed behavior, the five-backbone effective smoke effect is small but conservative (about +0.10 pp F1 mean) with very low aggregate true-crack mutation.

## Interpretation

1. Explicit KEEP supervision is technically viable and materially changes the formulation from pixel authenticity to corrective action learning.
2. Counterfactual over-segmentation can be mixed with natural FIT errors without using GT at runtime.
3. The new TCRR metric detects unsafe boundary removal that Dice/F1 alone can hide.
4. A core-only safety budget is insufficient; the parallel-boundary-strip guard reduces but does not eliminate true-crack removal.
5. VAL qualification is currently necessary and correctly forces most unstable policies to NO-OP.
6. V5.20 is not ready for full 3-seed/2-resolution qualification yet because only one of five seed-1337/128 REMOVE policies passes the stricter safety gate.

## Required next upgrades before full experiment

- calibrate counterfactual corruption distributions from real FIT Base-error statistics rather than fixed smoke parameters;
- add explicit region/component-level KEEP/REMOVE labels instead of relying only on per-pixel classification;
- add local orientation/width/branch/endpoint features;
- replace static boundary-strip heuristics with learned or calibrated component semantics;
- run Real-only vs Real+Counterfactual ablation on VAL before opening any further test evidence;
- only after ≥4/5 policies qualify on seed-1337 smoke should V5.20 expand to seeds 2027/31415 and 256×256;
- ADD/recovery remains a separate future integration stage and must not be replaced by an unvalidated heuristic.
