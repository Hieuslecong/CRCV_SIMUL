# CRCV V5.5c — Suppression Runtime Re-audit

## Decision

**P0 EVALUATION BUG FOUND AND CLOSED. CURRENT BEST SUPPRESSION = GT-FREE COMPONENT CONFIDENCE GATE. MULTI-BACKBONE QUALIFICATION STILL FAILS.**

The previous V5.5 suppression diagnostics used `ComponentBankDataset` during evaluation. That dataset is valid for conservative training labels, but it uses GT to exclude ambiguous components. Reusing it at runtime/evaluation made GT influence which Base components were eligible for deletion. Those earlier suppression numbers are therefore superseded for runtime claims.

V5.5c fixes the candidate-space equivalence issue by enumerating **all Base connected components without GT**. GT is used only after enumeration to measure CAL/evaluation metrics.

## 1. Corrected runtime candidate space

New module: `crcv52/suppression_v55c.py`

- `enumerate_runtime_components(record)` requires only `base` and `prob`.
- `confidence_suppression_mask(...)` requires no GT.
- `component_pixel_metrics(...)` uses GT only for CAL measurement after the runtime component set is fixed.
- CAL and runtime now operate on the same component space.

Regression status after the fix:

- full pytest: **36/36 PASS**
- focused V5.5/V5.5c tests: **12/12 PASS**
- compileall: PASS

## 2. Corrected Original-V52 suppression result

CAL-selected component-mean-probability gate, evaluated on **all** components:

- CAL true-pixel removal: **0.196%**
- CAL false-pixel removal: **45.879%**
- CAL gate: **PASS**
- threshold: **0.630070**

Development `real_val` diagnostic with the CAL threshold frozen:

- true-pixel removal: **0.759%**
- false-pixel removal: **49.503%**
- Dice delta: **+0.03266**
- clDice delta: **+0.10791**
- CC-error delta: **-11.5**
- precision delta: **+0.04054**
- recall delta: **-0.00370**
- normal predicted-pixel removal: **91.51%**
- normal connected-component reduction: **99.23%**

Because `real_val` has now been inspected repeatedly during V5.5 redesign, it is explicitly **retired as a fresh validation set** from V5.5c onward. It is development evidence only.

## 3. Does simulation improve suppression?

No reliable gain was found after the GT-free re-audit.

`prob_mean_only`:

- CAL FP removal 45.879%, TP removal 0.196%
- dev FP removal 49.503%, TP removal 0.759%
- dev Dice +0.03266, clDice +0.10791

`prob_mean + simulation-veto`:

- CAL FP removal 45.170%, TP removal 0.196%
- dev FP removal 49.447%, TP removal 0.759%
- dev Dice/clDice identical to the probability-only rule

Therefore the current evidence does **not** justify simulation as an active suppression signal. Forcing the XY prior into suppression would weaken the scientific story. Simulation remains justified for recovery geometry/proposal generation, where it encodes propagation plausibility.

## 4. Learned suppression ablation

A fully independent 3-seed learned suppressor (no parameter sharing with the failed recovery verifier) was also re-calibrated on all CAL components:

- CAL false-pixel removal: **54.946%**
- CAL true-pixel removal: **0.196%**
- dev false-pixel removal: **49.242%**
- dev true-pixel removal: **0.648%**
- dev Dice delta: **+0.02503**
- dev clDice delta: **+0.07604**

It removes slightly more CAL FP pixels but is worse than the much simpler probability gate on Dice/clDice, while adding 33,531 ensemble parameters. It is therefore not selected as the active suppression baseline.

## 5. Multi-backbone diagnostic

The same GT-free component-mean-probability method was calibrated independently on each CAL cache.

| Backbone | CAL TP removal | CAL FP removal | CAL gate | Dev Dice delta | Dev clDice delta |
|---|---:|---:|:---:|---:|---:|
| Original-V52 | 0.196% | 45.879% | PASS | +0.03266 | +0.10791 |
| DS-UNet-Lite | 0.922% | 2.372% | FAIL | +0.00259 | +0.01789 |
| Fast-SCNN-Lite | 0.388% | 0.201% | FAIL | -0.00106 | -0.00270 |
| BiSeNet-Tiny | 0.413% | 14.736% | FAIL | +0.01624 | +0.02958 |
| MobileNetV3-Small | 0.000% | 25.844% | FAIL | +0.00036 | +0.00232 |

Thus the strong Original-V52 result is **base-distribution specific** at this stage. The predeclared multi-backbone >=30% false-removal gate is not met.

The alternative backbones are diagnostic implementations in this workspace and several are weak/undertrained, so this table is not a definitive architecture ranking. It is nevertheless sufficient to block a backbone-agnostic suppression claim.

## 6. Scientific state after re-audit

```text
V5.4 proposal                     = QUALIFIED (frozen)
V5.5 relation verifier            = NOT QUALIFIED
V5.5c suppression Original-V52 CAL= PASS
V5.5c suppression multi-backbone  = FAIL
simulation for suppression        = NOT SUPPORTED / REJECTED
simulation for recovery geometry  = RETAINED
recovery_enabled                  = false
suppression_enabled               = false
runtime                           = FAIL_CLOSED_BASE_ONLY
final_test                        = SEALED_NOT_USED
```

## 7. Next justified work

1. Freeze V5.5c GT-free component enumeration as a correctness fix.
2. Treat component-mean-probability suppression as the **mandatory simple baseline**, not the paper novelty.
3. Do not claim simulation improves suppression unless a future clean experiment beats this baseline.
4. Focus novelty effort on simulation-informed recovery proposal + same-crack path verification.
5. Before any final claim, train/obtain properly qualified frozen alternative backbones and repeat suppression without using the retired `real_val` for tuning.
6. Keep the final test sealed until architecture/gates are frozen.
