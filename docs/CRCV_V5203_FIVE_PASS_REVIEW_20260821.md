# CRCV V5.20.3 — Five-Pass Engineering Review

Date: 2026-08-21  
Branch: `feature/crcv-v5-20-counterfactual-policy`  
Release status: **ENGINEERING_RC_PASS / MULTISEED_SCIENTIFIC_GATE_BLOCKED / Q1_BLOCKED**

## Scope

This review upgrades the V5.20.2 REMOVE subsystem and audits it five independent times before freezing an engineering release candidate. The opened `real_debug_data/test` split is development evidence only and is not a publication holdout.

## Pass 1 — Contracts and leakage

Findings and fixes:
- Action targets now require same-shape 2-D masks and assert exact Base/GT partitions.
- Runtime REMOVE feature/safety APIs remain GT-free.
- Legacy V5.20.1 keyword interfaces are retained as deprecated compatibility paths so existing tests/callers do not silently break.
- `ComponentRemovalConfig.max_pixels` remains accepted only as a deprecated explicit override; canonical V5.20.3 does not use it.

## Pass 2 — Numerical and scale correctness

Findings and fixes:
- Previous reference-distance normalization used the radius of the nearest reference boundary pixel, which is approximately one pixel and therefore was not truly scale invariant.
- Distance is now normalized by local half-width at the nearest reference skeleton point.
- Blur scale is normalized by image diagonal by default.
- Component policy now uses normalized area, skeleton length, bounding-box diagonal and elongation constraints.
- Error profiling is type-aware (`detached`, `attached_elongated`, `attached_shell_or_compact`) and exposes normalized geometry.

## Pass 3 — Topology and safety

Findings and fixes:
- A long boundary strip could be vetoed while a one-pixel terminal fragment near a crack endpoint remained removable.
- Endpoint/junction neighborhoods now receive local-radius protection.
- Safety projection now asserts that no Base skeleton pixel is removed.
- Existing foreground/image budgets and component-count checks remain fail-closed.

Important limitation: skeleton preservation and component-count preservation do not constitute complete topology preservation. Branch-level topology metrics are still required for publication claims.

## Pass 4 — Training and calibration reproducibility

Findings and fixes:
- V5.20.1 sampling seeds depended on record-list index, so manifest reordering could change the generated training corpus.
- V5.20.3 sorts FIT records by stable name and derives per-record/per-operator seeds from SHA-256 keyed identifiers.
- Duplicate record names and missing probability maps now fail closed.
- LightGBM canonical training uses deterministic settings and one worker by default.
- Training metadata records feature schema SHA-256, seed, Base threshold, simulator configuration and training configuration.

Rejected ablations:
- Promoting distal attached-FP pixels to canonical REMOVE supervision made real-data behavior too aggressive; canonical training keeps attached FP components as IGNORE and retains distal-pixel supervision only as an explicit ablation.
- Selecting operating points by effect/FPRR before TCRR produced CAL-aggressive policies that failed VAL safety. The safety-first selector was retained.

## Pass 5 — Regression and real-data end-to-end audit

Local review mirror:
- `python -m compileall`: PASS
- final focused regression suite: **25/25 PASS**
- deterministic training is invariant to FIT record ordering in the regression test.

### Seed 1337, 128×128

Across five Base backbones:
- ACTIVE: **4/5**
- mean deployed ΔDice: **+0.000834** (+0.0834 pp)
- mean deployed ΔCrack-IoU: **+0.000716**
- mean deployed ΔRecall: **-0.000460**
- mean TCRR: **0.000731** (0.0731%)
- mean FPRR: **0.005959**

### Seed 1337, 256×256

Across five Base backbones:
- ACTIVE: **4/5**
- mean deployed ΔDice: **+0.000108** (+0.0108 pp)
- mean deployed ΔCrack-IoU: **+0.000076**
- mean deployed ΔRecall: **-0.000405**
- mean TCRR: **0.000714** (0.0714%)
- mean FPRR: **0.001456**

Safety remains low-damage at both checked resolutions, but the 256 effect size is near zero and therefore is not evidence of robust performance improvement.

### Multi-seed 128×128 development gate

| Seed | ACTIVE / 5 | mean ΔDice | mean TCRR | Gate component |
|---:|---:|---:|---:|---|
| 1337 | 4/5 | +0.000834 | 0.000731 | pass |
| 2027 | 2/5 | +0.000128 | 0.000000 | **fail active-rate** |
| 31415 | 3/5 | +0.000341 | 0.000232 | **fail active-rate** |

Frozen development criterion requires at least 4/5 ACTIVE per seed. Therefore:

**MULTISEED_SCIENTIFIC_GATE = BLOCKED**

Failures:
- seed 2027: ACTIVE 2/5 < 4/5
- seed 31415: ACTIVE 3/5 < 4/5

The fail-closed runtime is behaving correctly: unsafe or non-beneficial policies revert to Base. However, the learned correction effect is not seed-robust enough to support a publication claim.

## Final engineering verdict

V5.20.3 is the strongest current **engineering release candidate** because it improves contracts, scale correctness, topology protection and deterministic retraining while preserving fail-closed behavior and backward compatibility.

It is **not** a final scientific/Q1 model. Remaining scientific blockers include:
1. multi-seed correction robustness;
2. small effect size;
3. no integrated ADD policy in the redesigned pipeline;
4. no LOBO backbone-transfer evidence;
5. no robust cross-dataset gain;
6. no fresh untouched external final holdout;
7. no direct published refinement comparator in the completed experiment;
8. no hierarchical/cluster statistical evidence for final claims;
9. incomplete structural/topology metrics.

The next scientific task should not loosen safety thresholds. It should improve the learned correction representation/generalization while keeping the V5.20.3 safety and reproducibility contracts frozen.
