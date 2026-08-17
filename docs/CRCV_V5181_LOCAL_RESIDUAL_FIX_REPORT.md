# CRCV V5.18.1 — Local Residual Suppression

Status: **development smoke**. Final/test remains sealed and was not read.

## Change

V5.18.1 retires whole terminal-branch deletion from the active suppression path and replaces it with **local residual suppression**:

```text
Frozen Base foreground
  -> foreground authenticity map
  -> very-low-authenticity pixels
  -> connected local residual regions
  -> strict region/total removal caps
  -> remove only suspicious subregions
  -> RGB width-aware ribbon recovery
  -> final mask
```

The runtime suppressor accepts no ground-truth input and guarantees that every removed pixel is a subset of the Frozen Base mask.

## Multi-backbone smoke

Three action-scorer seeds were evaluated: `1337`, `2027`, `31415`.

Mean absolute gains over Frozen Base across TinyUNet, Fast-SCNN-Lite, BiSeNet-Tiny, MobileNetV3-style segmentation and DS-UNet-Lite:

| Metric | Mean delta |
|---|---:|
| Precision | +4.367 pp |
| Recall | +4.819 pp |
| F1 | +4.611 pp |
| mIoU | +1.827 pp |

All five development backbones had positive Precision, Recall, F1 and mIoU. This is development evidence only; it is not an unseen-final or zero-shot-backbone generalization claim.

## Safety invariants

- runtime API has no GT argument;
- removed pixels are always a subset of Frozen Base foreground;
- per-region and total-removal caps are fail-closed;
- high-authenticity foreground is preserved;
- final/test split remains sealed.

## Files in this branch

- `crcv52/local_residual_suppression.py` — production-style runtime suppressor;
- `tests/test_local_residual_suppression_v5181.py` — safety/property tests.

The next qualification step is a fresh CAL/VAL split and explicit leave-one-backbone-out testing before any model-agnostic claim.
