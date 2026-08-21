# CRCV Q1 redesign — smoke qualification report (2026-08-21)

Branch: `feature/crcv-q1-redesign-v2`

## Decision

**Engineering smoke: PASS. Scientific retraining smoke: BLOCKED by missing real data/checkpoints.**

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

## 6. Bundle integrity smoke

`sha256sum -c SHA256SUMS.txt` was executed for the supplied audit bundle.

Result: **all entries PASS**.

## 7. Historical split-guard smoke

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

## 8. Full real-data V5.17 → V5.18 → V5.18.1 smoke

Attempted entrypoint execution for seeds `1337`, `2027`, `31415` and resolutions `128`, `256`.

Status: **BLOCKED BEFORE TRAINING**.

Missing inputs include:

- `/mnt/data/v516_data/real_debug_data/manifest.csv` and referenced raw Images/Labels;
- `/mnt/data/v516_data/debug_pack/dataset/Images`;
- `/mnt/data/v516_data/debug_pack/dataset/Labels`;
- `/mnt/data/v517_work/full/v54_release_full/artifacts/models/geometry_xy.pt`;
- `/mnt/data/v517_work/full/v54_release_full/artifacts/models/centerline_field_v52b.pt`;
- `/mnt/data/v517_work/full/v54_release_full/artifacts/models/endpoint_ranker_v53.pkl`;
- downstream `v517_banks.pkl`, `add_variants_partial.pkl`, `foreground_authenticity.pkl` for resume workflows.

The supplied audit bundle explicitly reports that `v517_banks.pkl`, `add_variants_partial.pkl`, `foreground_authenticity.pkl` and the older width-suppression smoke script are not bundled.

The full-seed scripts therefore cannot produce a legitimate post-redesign F1/Dice/IoU result in the current execution environment.

## 9. Interpretation

The redesign is **engineering-smoke qualified** for the tested components. No crash, NaN/Inf, output-shape, gradient or basic evidence-gate regression was detected in the executable smoke coverage.

However, the project is **not scientifically smoke-qualified yet**. In particular, the following remain unmeasured after the suppression redesign:

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

## 10. Required next input to unblock real smoke

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
