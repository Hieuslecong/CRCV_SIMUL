# CRCV V5.5 Trial Training Report — 2026-08-16

> **SUPERSEDED FOR SUPPRESSION RUNTIME CLAIMS BY V5.5c P0 RE-AUDIT.** The suppression sections below used `ComponentBankDataset` during evaluation. That dataset excludes GT-ambiguous components and therefore let GT influence runtime component eligibility. Recovery results remain valid; suppression numbers below are retained only as historical development evidence. See `P0_SUPPRESSION_RUNTIME_REAUDIT.md` for corrected all-component results.

## Decision

**Suppression showed an initial positive signal; recovery verification still fails. Full V5.5 remains fail-closed.**

No final-test sample was used. V5.4 proposal generation stayed frozen.

## 1. Recovery verifier

Three recovery formulations were inspected on CAL.

- Direct relational verifier: AUC 0.7127, AP 0.0082; exact AddedPrecision 17.57%; CoreGap recovery 2.86%; normal added pixels 144.
- Controlled-gap curriculum + natural fine-tuning: AUC 0.7140, AP 0.0065; AddedPrecision 75.00%; CoreGap recovery 0.00%; normal added pixels 26.
- Path-validity redesign: best candidate-level discrimination improved to about AUC 0.792 / AP 0.032, but under the required zero-normal condition the best exact operating point remained only ~16.7% precision / 1.43% CoreGap recovery.

**Recovery gate: FAIL.** No recovery output may affect runtime.

## 2. Historical suppression CAL training — superseded

The simulation-aware component suppressor was trained for 20 epochs with three seeds, always using the CAL-derived threshold only.

| Seed | True-pixel removal | False-pixel removal | CAL gate |
|---:|---:|---:|:---:|
| 5551 | 0.095% | 31.11% | PASS |
| 5552 | 0.095% | 31.58% | PASS |
| 5553 | 0.095% | 31.75% | PASS |

These values are not valid for runtime qualification because the evaluation candidate set was GT-filtered. They are superseded by V5.5c.

## 3. Historical real_val diagnostic — superseded

Across the same three seeds the earlier diagnostic reported true-pixel removal 0.232% ± 0.047%, false-pixel removal 22.13% ± 0.66%, Dice +0.01531 ± 0.00012 and clDice +0.05221 ± 0.00175. These values are retained only for provenance and must not be used for runtime/paper claims.

## 4. Correct scientific state after V5.5c re-audit

```text
proposal_qualified          = true
relation_verifier_qualified = false
suppression_original_CAL    = PASS with GT-free all-component confidence baseline
suppression_multibackbone   = false
simulation_for_suppression  = rejected by current evidence
recovery_enabled            = false
suppression_enabled         = false
final_test                  = SEALED_NOT_USED
runtime                     = FAIL_CLOSED_BASE_ONLY
```

The authoritative suppression report is `P0_SUPPRESSION_RUNTIME_REAUDIT.md`.
