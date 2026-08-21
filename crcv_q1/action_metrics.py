from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def _mask2d(mask, name: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    return arr


def correction_metrics(base_mask, refined_mask, gt_mask) -> dict[str, float]:
    base = _mask2d(base_mask, "base_mask")
    refined = _mask2d(refined_mask, "refined_mask")
    gt = _mask2d(gt_mask, "gt_mask")
    if base.shape != refined.shape or base.shape != gt.shape:
        raise ValueError("all masks must have identical shapes")
    removed = base & ~refined
    base_tp = base & gt
    base_fp = base & ~gt
    true_removed = removed & gt
    fp_removed = removed & ~gt
    tcrr = true_removed.sum() / max(int(base_tp.sum()), 1)
    fprr = fp_removed.sum() / max(int(base_fp.sum()), 1)
    base_skeleton = skeletonize(base)
    skeleton_removed = int((base_skeleton & ~refined).sum())
    return {
        "removed_pixels": float(removed.sum()),
        "tcrr": float(tcrr),
        "fprr": float(fprr),
        "true_crack_removed": float(true_removed.sum()),
        "false_positive_removed": float(fp_removed.sum()),
        "base_skeleton_removed": float(skeleton_removed),
    }


def binary_segmentation_metrics(pred_mask, gt_mask) -> dict[str, float]:
    p = _mask2d(pred_mask, "pred_mask")
    g = _mask2d(gt_mask, "gt_mask")
    if p.shape != g.shape:
        raise ValueError("pred_mask and gt_mask must match")
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(f1),
        "crack_iou": float(iou),
    }
