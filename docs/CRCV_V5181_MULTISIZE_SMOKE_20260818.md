# CRCV V5.18.1 multi-resolution smoke — 2026-08-18

## Protocol

This experiment evaluates the V5.18-family refinement block at two native training/evaluation resolutions: **128x128** and **256x256**.

For each resolution the intended protocol is:

1. train each Base model again at the target resolution;
2. select Base threshold on CAL only;
3. regenerate recovery candidates at the same resolution;
4. retrain the RGB width-aware ribbon refiner;
5. retrain foreground-authenticity / local residual suppression;
6. retrain the metric-aligned ADD scorer;
7. select CRCV thresholds on CAL only;
8. evaluate on VAL;
9. keep final/test sealed.

No 160px proposal bank or mask is resized and reused as a substitute for a native-resolution run.

Backbones:

- TinyUNet
- Fast-SCNN-Lite
- BiSeNet-Tiny
- MobileNetV3-style segmentation
- DS-UNet-Lite

Action-scorer seeds: `1337`, `2027`, `31415`.

## 128x128 completed result

Mean Base -> CRCV gain over the five backbones:

- Precision: **+1.567 pp**
- Recall: **+4.120 pp**
- F1: **+3.550 pp**
- mIoU: **+1.500 pp**

Per-backbone approximate gain:

| Backbone | Delta F1 | Delta mIoU |
|---|---:|---:|
| TinyUNet | +1.09 pp | +0.51 pp |
| Fast-SCNN-Lite | +6.72 pp | +2.78 pp |
| BiSeNet-Tiny | +5.09 pp | +2.04 pp |
| MobileNetV3-style | +2.78 pp | +1.23 pp |
| DS-UNet-Lite | +2.07 pp | +0.95 pp |

The 128px learned RGB ribbon refiner was retrained at that resolution using **120,177 pixel samples from 5,540 candidate corridors**. CAL selected ribbon threshold `0.55`.

All 5/5 backbones improved in F1 and mIoU at 128px. TinyUNet showed a small precision decrease in repeated scorer seeds, so the result should not be summarized as all four metrics improving for every backbone.

## 256x256 status

The 256px run is **not complete yet** and must not be reported as a CRCV result.

Base training had completed for four backbones when the execution window ended. Available Base VAL F1 values were:

| Backbone | Base F1 @128 | Base F1 @256 |
|---|---:|---:|
| TinyUNet | 0.5276 | 0.5853 |
| Fast-SCNN-Lite | 0.3953 | 0.4645 |
| BiSeNet-Tiny | 0.3904 | 0.4437 |
| MobileNetV3-style | 0.4478 | 0.4809 |

DS-UNet-Lite @256 and the full V5.18.1 refinement pipeline at 256 remain pending.

Therefore no Delta F1 / Delta mIoU claim for 256 is valid yet.

## Current interpretation

The completed 128px run supports the resolution robustness of the V5.18-family refinement block at development-smoke level. A valid 128 vs 256 conclusion requires completion of the full native 256 pipeline, including candidate/ribbon regeneration and CAL-only threshold selection.

Final/test remains **SEALED / NOT READ**.
