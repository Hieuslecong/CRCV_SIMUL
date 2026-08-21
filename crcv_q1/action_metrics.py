from __future__ import annotations

import numpy as np


def correction_metrics(base_mask, refined_mask, gt_mask) -> dict[str, float]:
    base = np.asarray(base_mask, dtype=bool)
    refined = np.asarray(refined_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if base.shape != refined.shape or base.shape != gt.shape:
        raise ValueError("all masks must have identical shapes")
    removed = base & ~refined
    base_tp = base & gt
    base_fp = base & ~gt
    true_removed = removed & gt
    fp_removed = removed & ~gt
    tcrr = true_removed.sum() / max(int(base_tp.sum()), 1)
    fprr = fp_removed.sum() / max(int(base_fp.sum()), 1)
    return {
        "removed_pixels": float(removed.sum()),
        "tcrr": float(tcrr),
        "fprr": float(fprr),
        "true_crack_removed": float(true_removed.sum()),
        "false_positive_removed": float(fp_removed.sum()),
    }


def binary_segmentation_metrics(pred_mask, gt_mask) -> dict[str, float]:
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": float(precision), "recall": float(recall), "dice": float(f1), "crack_iou": float(iou)}
