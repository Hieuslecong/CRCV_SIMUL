# CRCV Q1 redesign — smoke qualification report (2026-08-21)

Branch: `feature/crcv-q1-redesign-v2`

## Decision

**Engineering smoke: PASS. Scientific retraining smoke: BLOCKED by unavailable real-data/upstream assets in the active execution environment.**

This report deliberately separates structural/synthetic smoke tests from publication evidence. Synthetic data are used only to detect crashes, shape errors, non-finite losses/gradients and unsafe suppression behavior. They are not used for accuracy claims.

## 1. Compile smoke

`python -m compileall` was run over the reconstructed redesign smoke modules and the bundled V5.17/V5.18/V5.18.1 scripts.

Result: **PASS**.

## 2. Q1 protocol / regression smoke

The post-redesign protocol, evidence, split-guard, statistics and local-suppression tests were executed from the exact branch contents reconstructed in the execution environment.

Result: **14/14 tests PASS**.

Covered checks include:

- protocol hash stability;
- cluster-aware bootstrap/permutation utilities;
- Holm bounds;
- evidence package pass/fail behavior;
- mandatory published refiner;
- self-declared LOBO cannot pass;
- artifact SHA mismatch blocks evidence;
- historically exposed final sample is rejected;
- removed pixels remain inside Frozen Base;
- high-authenticity crack preservation;
- large-region deletion cap;
- sparse-foreground deletion budget;
- empty-Base safety;
- runtime suppression API refuses a GT argument.

## 3. Base-network forward/backward training smoke

A structural synthetic training smoke was executed for the five development backbones extracted from `v517_fullseed.py`:

- TinyUNet
- FastSCNNLite
- BiSeNetTiny
- MobileNetV3SmallSeg
- DSUNetLite

Matrix:

- resolutions: 128, 256;
- seeds: 1337, 2027, 31415;
- backbones: 5;
- total cases: 30.

Each case performed forward, segmentation loss, backward, gradient check and optimizer steps.

Result: **30/30 PASS**.

All cases had:

- correct output shape;
- finite logits/loss;
- non-zero finite gradients;
- successful optimizer update.

Aggregate structural-smoke values (not scientific metrics):

| Resolution | Cases | Pass | Mean loss step 0 | Mean loss step 1 |
|---|---:|---:|---:|---:|
| 128 | 15 | 15 | 1.8140 | 1.7844 |
| 256 | 15 | 15 | 1.7718 | 1.7536 |

## 4. ADD/REMOVE action-scorer smoke

LightGBM classifier/regressor structural smoke was executed for:

- ADD × seeds 1337/2027/31415;
- REMOVE × seeds 1337/2027/31415.

Result: **6/6 PASS**.

All predictions were finite and classifier probabilities stayed in `[0,1]`.

This test validates the basic training/inference code path only; synthetic scorer accuracy/RMSE are not research results.

## 5. Local residual suppression differential smoke

The old image-area-only suppression budget was compared with the redesigned dual budget on 200 synthetic crack-like masks:

- 100 cases at 128;
- 100 cases at 256.

Results:

| Resolution | New mask always subset of old removal | Always inside Base | Max removed fraction of foreground | More conservative cases |
|---|---|---|---:|---:|
| 128 | yes | yes | 17.51% | 5/100 |
| 256 | yes | yes | 19.94% | 12/100 |

The redesign never removed pixels outside the old eligible removal set in this smoke and respected the 20% foreground budget.

## 6. Supplied foreground-authenticity artifact smoke

A separately supplied `foreground_authenticity.pkl` was inspected before loading and then loaded through a restricted allow-list unpickler.

Artifact metadata:

- SHA256: `5150a08891b9b62d4472509344eb97139dc3c286222c1b0bc30e3712211470b2`;
- size: 362,019 bytes;
- pickle protocol: 4;
- model: `lightgbm.sklearn.LGBMClassifier`;
- features: 13;
- estimators: 200;
- classes: `[0, 1]`;
- `random_state=1856`.

`v518_fullseed.py` constructs this model with `random_state=SEED+519`, therefore the supplied artifact is configuration-consistent with **CRCV full seed 1337**.

A structural compatibility smoke rebuilt the exact 13-feature foreground-authenticity contract and evaluated the artifact on synthetic crack-like records at 128 and 256 for input seeds 1337/2027/31415.

Result: **6/6 PASS**.

All cases produced finite probabilities in `[0,1]`, valid output shapes, and suppression removed pixels only inside Frozen Base. This is compatibility evidence only, not segmentation-performance evidence.

Provenance record committed at:

`artifacts/provenance/foreground_authenticity_seed1337.json`

Important limitation: the supplied pickle proves model/configuration compatibility, but does **not** by itself prove the training-dataset lineage or absence of historical evaluation-set exposure.

## 7. Bundle integrity smoke

`sha256sum -c SHA256SUMS.txt` was executed for the supplied audit bundle.

Result: **all entries PASS**.

## 8. Historical split-guard smoke

`split_assignment.csv` contains 320 rows. Treating the historical `test` role as exposed and attempting to reinterpret the current test as a fresh final holdout produced:

- status: **FAIL**;
- current final/test samples marked historically exposed: 88;
- source-family overlap warning between development and final pools.

Historical transition counts:

| old split | new test | new train | new val |
|---|---:|---:|---:|
| test | 13 | 29 | 8 |
| train | 58 | 123 | 29 |
| val | 17 | 30 | 13 |

Therefore the supplied current test split must **not** be certified as `SEALED_FRESH_EXTERNAL`.

## 9. Full real-data V5.17 → V5.18 → V5.18.1 smoke

Attempted entrypoint execution for seeds `1337`, `2027`, `31415` and resolutions `128`, `256`.

Status in the active execution environment: **BLOCKED BEFORE TRAINING**.

The newly supplied `foreground_authenticity.pkl` is no longer a missing dependency for a seed-1337 resume path. However, the active `/mnt/data` workspace still does not expose the real-data tree and upstream proposal assets referenced by the bundled full-seed scripts, including paths equivalent to:

- `real_debug_data/manifest.csv` and referenced raw Images/Labels;
- pretrain/debug `Images` and `Labels`;
- `geometry_xy.pt`;
- `centerline_field_v52b.pt`;
- `endpoint_ranker_v53.pkl`.

If these assets exist on the user's training machine but are simply not attached/mounted in this conversation runtime, this is an **environment-availability blocker**, not evidence that the project itself lacks them.

The full-seed scripts therefore cannot produce a legitimate post-redesign F1/Dice/IoU result inside the current execution environment yet.

## 10. Interpretation

The redesign is **engineering-smoke qualified** for the tested components. No crash, NaN/Inf, output-shape, gradient or basic evidence-gate regression was detected in the executable smoke coverage. The supplied seed-1337 foreground-authenticity model is structurally compatible with the V5.18 feature contract and the redesigned suppression runtime.

However, the project is **not scientifically smoke-qualified yet** in this runtime. In particular, the following remain unmeasured after the suppression redesign:

- real-data 128 CRCV delta;
- native 256 CRCV delta;
- full-seed 1337/2027/31415 variability;
- ADD/REMOVE interaction on real data;
- Crack-IoU/Dice after redesign;
- full-pipeline GT mutation invariance;
- published-refiner comparison;
- LOBO;
- cross-dataset routes.

Existing V5.18.1 result artifacts remain historical development evidence only and must not be reported as the performance of the redesigned suppression code.

## 11. Required next execution step

Before running the full real-data smoke, run:

```bash
python scripts/preflight_fullseed_assets.py \
  --data-root /path/to/v516_data \
  --proposal-root /path/to/v54_release_full \
  --v517-module /path/to/v517_fullseed.py \
  --v518-module /path/to/v518_fullseed.py \
  --run-root /path/to/run_root
```

The preflight must return `PASS` before GPU/full scientific smoke is considered runnable.
