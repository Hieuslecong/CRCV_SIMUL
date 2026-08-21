from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ComponentRemovalConfig:
    """Scale-free conservative whole-component REMOVE fallback.

    `max_pixels` is a deprecated compatibility override. Canonical V5.20.3 leaves
    it as None and uses normalized shape/area constraints only.
    """
    max_area_fraction: float = 0.001
    max_mean_probability: float = 0.85
    max_skeleton_length_fraction: float = 0.025
    max_bbox_diagonal_fraction: float = 0.04
    max_elongation: float = 4.0
    max_total_remove_fraction: float = 0.03
    max_foreground_remove_fraction: float = 0.10
    max_pixels: int | None = None


def _elongation(ys: np.ndarray, xs: np.ndarray) -> float:
    if len(ys) < 3:
        return 1.0
    pts = np.stack([ys, xs], axis=1).astype(np.float32)
    vals = np.maximum(np.linalg.eigvalsh(np.cov(pts, rowvar=False)), 1e-6)
    return float(np.sqrt(vals[-1] / vals[0]))


def component_remove_mask(base_mask, probability,
                          config: ComponentRemovalConfig | None = None) -> np.ndarray:
    """Return a GT-free, shape-aware whole-component REMOVE mask.

    Long/elongated crack-like components are protected even when small in area.
    Canonical size thresholds are normalized by image dimensions. The deprecated
    `max_pixels` override is honored only when an old caller explicitly sets it.
    """
    cfg = config or ComponentRemovalConfig()
    base = np.asarray(base_mask, bool)
    prob = np.asarray(probability, np.float32)
    if base.shape != prob.shape or base.ndim != 2:
        raise ValueError("bad base/probability shapes")
    if not np.isfinite(prob).all():
        raise ValueError("probability contains non-finite values")
    if not base.any():
        return np.zeros_like(base)
    if not (0 <= cfg.max_area_fraction <= 1 and
            0 <= cfg.max_total_remove_fraction <= 1 and
            0 <= cfg.max_foreground_remove_fraction <= 1):
        raise ValueError("fraction limits must be in [0,1]")
    if cfg.max_pixels is not None and int(cfg.max_pixels) < 0:
        raise ValueError("max_pixels must be non-negative or None")

    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    labels, n = ndi.label(base)
    candidates = []
    for lab in range(1, n + 1):
        reg = labels == lab
        ys, xs = np.where(reg)
        size = len(ys)
        if size == 0:
            continue
        area_fraction = size / float(base.size)
        mean_probability = float(prob[reg].mean())
        sk_len_fraction = float(skeletonize(reg).sum()) / diag
        bbox_diag_fraction = float(np.hypot(ys.max()-ys.min()+1,
                                            xs.max()-xs.min()+1)) / diag
        elong = _elongation(ys, xs)
        if cfg.max_pixels is not None and size > int(cfg.max_pixels):
            continue
        if area_fraction > cfg.max_area_fraction:
            continue
        if mean_probability > cfg.max_mean_probability:
            continue
        if sk_len_fraction > cfg.max_skeleton_length_fraction:
            continue
        if bbox_diag_fraction > cfg.max_bbox_diagonal_fraction:
            continue
        if elong > cfg.max_elongation:
            continue
        candidates.append((mean_probability, area_fraction, bbox_diag_fraction, lab, reg))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    image_budget = int(np.floor(cfg.max_total_remove_fraction * base.size))
    foreground_budget = int(np.floor(cfg.max_foreground_remove_fraction * int(base.sum())))
    budget = max(0, min(image_budget, foreground_budget))
    if budget <= 0:
        return np.zeros_like(base)

    removed = np.zeros_like(base)
    used = 0
    for _, _, _, _, reg in candidates:
        size = int(reg.sum())
        if used + size > budget:
            continue
        removed |= reg
        used += size
    return removed
