# CRCV V5.6/V5.6b Recovery Redesign Trial

## Decision

**RECOVERY STILL NOT QUALIFIED.** Two cleaner recovery formulations were tested after the V5.5c suppression re-audit. Neither reaches the predeclared recovery gate, so recovery remains OFF and the final test remains sealed.

## 1. Source gate discovered from Module-TRAIN only

The frozen V5.4 source score is highly informative at source-group level. A threshold chosen only from Module-TRAIN to retain 100% of Module useful source groups is:

- source gate threshold: **0.7223853**
- Module candidates after gate: ~3.9k instead of ~23k
- CAL candidates after the same frozen gate: **3,671** instead of ~22.7k
- CAL useful source recall: **4/4**

This source gate is retained as a valid efficiency/safety idea because it is selected before CAL acceptance tuning and does not use GT at runtime.

## 2. V5.6 — source-gated utility ranker

Change from V5.5: the main ranking target is the actual useful recovery label (high exact path precision + CoreGap contribution), instead of generic path validity. The frozen source gate is applied before relational scoring.

Seed 5607, 8 natural Module-TRAIN epochs:

- CAL candidate AUC: **0.83916**
- CAL AP: **0.07846**
- best zero-normal operating point:
  - AddedPrecision: **15.28%**
  - CoreGap recovery: **0.00%**
  - TP/FP: **11/61**
  - normal added: **0**

**Gate: FAIL.** Candidate discrimination improved strongly, but the highest-scoring paths are often high-confidence non-recovery paths. AUC/AP improvement does not translate into safe topological recovery.

## 3. V5.6b — OOF controlled curriculum + natural utility fine-tuning

To increase diversity without touching CAL/final test, controlled gaps were generated from the full frozen OOF FIT cache:

- controlled samples: **1,326**
- controlled same-source groups: **309**
- natural Module candidates after source gate: **3,882**
- natural useful positives: **421**
- natural source groups: **29**

Training sequence:

1. 5 epochs controlled OOF curriculum on real RGB;
2. 8 epochs natural Module-TRAIN utility ranking;
3. CAL used only for final operating-point calibration.

Seed 5611 result:

- CAL AUC: **0.59353**
- CAL AP: **0.02283**
- best zero-normal operating point:
  - AddedPrecision: **100%**
  - CoreGap recovery: **2.86%**
  - TP/FP: **4/0**
  - Core hit: **2/70**
  - normal added: **0**

**Gate: FAIL.** The controlled curriculum improves safety for one accepted path but introduces a substantial domain gap and does not provide sufficient natural recovery coverage.

## 4. Failure analysis

The CAL useful recovery events are extremely sparse. In the frozen bank, only four CAL source groups contain candidates satisfying the strict V5.4 useful-recovery label. The failure is not proposal coverage; V5.4 proposal remains qualified. The failure is ranking the correct path inside a source group and rejecting visually plausible alternatives.

Examples show two recurrent errors:

- high-ridge/texture paths can look more crack-like than the correct trajectory but have zero GT overlap;
- long ridge continuations can receive very high verifier confidence while a shorter precise V5.2b path is the correct repair.

A hard length analysis is stable across Module/CAL: all useful positive candidates are short (Module max 17 px; CAL max 18 px), whereas negatives extend to 72 px. This is useful as a predeclared structural safety feature, but length alone cannot solve short hard-negative cases.

## 5. Implementation/performance correction

During V5.6 experimentation, candidate crop construction was audited. Controlled trajectories can leave image bounds, so crop centers must be clamped before extracting fixed-size views. Sparse V5.4 candidate masks are already centerline-like; avoiding full-image skeletonization for <=128-pixel candidates makes candidate reconstruction much faster without changing their graph route. Local regression remains **36/36 PASS** and `compileall` PASS.

## 6. Scientific state

```text
V5.4 proposal                     = QUALIFIED / FROZEN
V5.5c suppression Original-V52   = CAL PASS, not multi-backbone qualified
simulation for suppression        = NOT SUPPORTED
simulation for recovery geometry  = RETAINED
V5.6 source-gated utility ranker  = FAIL
V5.6b controlled+natural ranker   = FAIL
recovery_enabled                  = false
suppression_enabled               = false
runtime                           = FAIL_CLOSED_BASE_ONLY
final_test                        = SEALED_NOT_USED
```

## 7. Next justified redesign

Do not add more proposal families. Freeze V5.4 geometry. The next verifier should explicitly preserve the ordered path rather than resize the whole trajectory into one square crop. The evidence-backed next architecture is a **path-aligned strip verifier**:

- sample local RGB/ridge/gradient/probability cross-sections along the ordered candidate path;
- compare those sequential features with source crack context;
- include the frozen simulation geometry score as a prior, not as RGB evidence;
- train with same-source listwise competition and an abstention head;
- retain the Module-derived source gate and a short-path structural safety bound;
- use CAL once for acceptance thresholds only.

This redesign directly targets the observed hard negatives: locally crack-like texture that fails continuity along the complete candidate trajectory.
