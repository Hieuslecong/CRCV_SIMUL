# CRCV V5.5 Trial Training Report — 2026-08-16

## Decision

**Suppression shows a robust positive signal; recovery verification still fails. Full V5.5 remains fail-closed.**

No final-test sample was used. V5.4 proposal generation stayed frozen.

## 1. Recovery verifier

Three recovery formulations were inspected on CAL.

- Direct relational verifier: AUC 0.7127, AP 0.0082; exact AddedPrecision 17.57%; CoreGap recovery 2.86%; normal added pixels 144.
- Controlled-gap curriculum + natural fine-tuning: AUC 0.7140, AP 0.0065; AddedPrecision 75.00%; CoreGap recovery 0.00%; normal added pixels 26.
- Path-validity redesign: best candidate-level discrimination improved to about AUC 0.792 / AP 0.032, but under the required zero-normal condition the best exact operating point remained only ~16.7% precision / 1.43% CoreGap recovery.

**Recovery gate: FAIL.** No recovery output may affect runtime.

## 2. Suppression CAL training

The simulation-aware component suppressor was trained for 20 epochs with three seeds, always using the CAL-derived threshold only.

| Seed | True-pixel removal | False-pixel removal | CAL gate |
|---:|---:|---:|:---:|
| 5551 | 0.095% | 31.11% | PASS |
| 5552 | 0.095% | 31.58% | PASS |
| 5553 | 0.095% | 31.75% | PASS |

Mean CAL false-pixel removal: **31.48%**. Mean true-pixel removal: **0.095%**. All three seeds pass the CAL suppression gate (TP removal <=1%, FP removal >=30%).

## 3. External real_val diagnostic (threshold frozen from CAL)

Across the same three seeds:

- true-pixel removal: **0.232% ± 0.047%**
- false-pixel removal: **22.13% ± 0.66%**
- crack Dice delta: **+0.01531 ± 0.00012**
- crack clDice delta: **+0.05221 ± 0.00175**
- CC-error delta: **-8.90 ± 0.26**
- precision delta: **+0.01932 ± 0.00015**
- recall delta: **-0.00069 ± 0.00034**
- normal predicted-pixel removal: **41.65% ± 1.49%**
- normal connected-component reduction: **78.29% ± 0.65%**

The external signal is strong and very seed-stable for Dice/clDice/topology, but false-pixel removal is only ~22.1%, below the predeclared >=30% external target. Therefore suppression is **promising but not runtime-qualified**.

## 4. Current scientific state

```text
proposal_qualified          = true
sim_prior_profile_fitted    = true
relation_verifier_qualified = false
suppression_CAL_gate        = true (3/3 seeds)
suppression_external_gate   = false
recovery_enabled            = false
suppression_enabled         = false
final_test                  = SEALED_NOT_USED
runtime                     = FAIL_CLOSED_BASE_ONLY
```

## 5. Interpretation

The trial changes the priority: the simulation-aware suppression branch now has reproducible evidence of improving segmentation and topology while preserving crack recall. The recovery branch remains the blocker and should not be activated. The next justified experiment is to validate suppression on additional frozen backbones without tuning its threshold on real_val, while redesigning recovery acceptance separately.
