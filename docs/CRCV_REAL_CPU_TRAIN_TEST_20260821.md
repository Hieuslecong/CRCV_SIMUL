# CRCV real-data CPU training and test report — 2026-08-21

## Status

**Base segmentation real-data training: COMPLETE.**
**REMOVE-v2 real-data qualification: COMPLETE for the available pipeline.**
**Full CRCV ADD+REMOVE: NOT RUN because the proposal/checkpoint assets required by ADD are not available in this runtime.**

The user explicitly requested TEST evaluation. Therefore the evaluated `real_debug_data/test` split is now **development evidence**, not a sealed final publication holdout.

## Data used

### real_debug_data
- train: 60 images
- deterministic stratified split of train: FIT=48, CAL=12
- VAL: 20 images
- TEST: 10 crack images
- source resolution: 256×256

### debug_pack
- pretraining train list: 150 images
- validation list: 40 images
- test list: 40 images
- `debug_pack/test` is used only as a secondary robustness diagnostic; thresholds are not tuned on it.

## CPU training protocol

- CPU only (`torch 2.10.0+cpu`, CUDA disabled)
- seeds: 1337, 2027, 31415
- resolutions: 128 and 256
- backbones: TinyUNet, FastSCNNLite, BiSeNetTiny, MobileNetV3SmallSeg, DSUNetLite
- exact V5.17 schedule: 2 pretrain epochs + 7 fine-tune epochs
- Base threshold selected on CAL only
- total Base trainings: 3 × 2 × 5 = **30 real-data models**

## Main real_debug_data TEST result — across seeds and five backbones

| Resolution | Base F1 | REMOVE-v2 F1 | ΔF1 | Base Crack-IoU | REMOVE-v2 Crack-IoU | ΔIoU | REMOVE active seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.3940 ± 0.0161 | 0.4065 ± 0.0137 | +0.0125 | 0.2520 ± 0.0126 | 0.2618 ± 0.0126 | +0.0098 | 2/3 |
| 256 | 0.3644 ± 0.0371 | 0.3656 ± 0.0363 | +0.0013 | 0.2348 ± 0.0275 | 0.2358 ± 0.0268 | +0.0010 | 1/3 |

The gain values include fail-closed seeds as Δ=0, which is the correct deployed behavior.

## REMOVE-v2 calibration outcome by seed

- 1337 @128: ACTIVE, threshold 0.040
- 1337 @256: ACTIVE, threshold 0.015
- 2027 @128: NO SAFE CAL THRESHOLD → NO-OP
- 2027 @256: NO SAFE CAL THRESHOLD → NO-OP
- 31415 @128: ACTIVE, threshold 0.040
- 31415 @256: NO SAFE CAL THRESHOLD → NO-OP

Thus REMOVE-v2 is **not seed-robust yet**. The fail-closed mechanism works, but half of the seed×resolution configurations cannot safely suppress under the frozen CAL rule.

## Secondary debug_pack/test robustness diagnostic

| Resolution | Base F1 | REMOVE-v2 F1 | ΔF1 | Base Crack-IoU | REMOVE-v2 Crack-IoU | ΔIoU |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 0.4713 ± 0.0195 | 0.4702 ± 0.0211 | -0.0012 | 0.3126 | 0.3117 | -0.0009 |
| 256 | 0.4717 ± 0.0174 | 0.4713 ± 0.0175 | -0.0003 | 0.3124 | 0.3121 | -0.0003 |

This diagnostic is slightly negative for F1/Crack-IoU after REMOVE. It does **not** support a cross-domain/generalization claim.

## Scientific interpretation

1. The real-data CPU training pipeline is executable and completes all 30 Base configurations.
2. The current panel is seed-sensitive. MobileNetV3SmallSeg and some 256 runs show especially large seed variation.
3. 128 currently outperforms 256 on the five-backbone mean, so the current implementation is scale-sensitive.
4. REMOVE-v2 provides positive in-pool TEST effects when its CAL gate activates, and fail-closes when unsafe.
5. REMOVE-v2 does not yet show robust transfer to `debug_pack/test`.
6. These numbers evaluate **Base + REMOVE-v2**, not the full CRCV method. No ADD/recovery claim can be made from this run.
7. A fresh external final holdout is still required for publication evidence.

## Missing assets for full CRCV ADD+REMOVE

- `geometry_xy.pt`
- `centerline_field_v52b.pt`
- `endpoint_ranker_v53.pkl`
- `v517_banks.pkl` or enough inputs to regenerate it
- `add_variants_partial.pkl` or enough inputs to regenerate it

Until these are available, replacing ADD with a heuristic would invalidate comparison with CRCV V5.18.1.
