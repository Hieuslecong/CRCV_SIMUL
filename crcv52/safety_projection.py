from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class SafetyProjectionConfig:
    core_protection_radius: int = 1
    max_total_remove_fraction: float = 0.03
    max_foreground_remove_fraction: float = 0.10
    max_component_fraction: float = 0.02
    preserve_component_count: bool = True


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    disk = (xx*xx + yy*yy) <= r*r
    return ndi.binary_dilation(mask, structure=disk)


def project_remove(base_mask, remove_score, threshold: float,
                   config: SafetyProjectionConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project REMOVE scores into a conservative, GT-free corrected mask.

    Runtime inputs are only Base and REMOVE score. The skeleton-derived protected
    support is computed from Base itself, so no GT is accepted or required.
    """
    cfg = config or SafetyProjectionConfig()
    base = np.asarray(base_mask, dtype=bool)
    score = np.asarray(remove_score, dtype=float)
    if base.shape != score.shape or base.ndim != 2:
        raise ValueError("base_mask and remove_score must be same-shape 2-D arrays")
    if not np.isfinite(score).all():
        raise ValueError("remove_score contains non-finite values")
    if not base.any():
        return base.copy(), np.zeros_like(base), {"status":"NO_OP_EMPTY_BASE"}

    sk = skeletonize(base)
    protected = _dilate(sk, cfg.core_protection_radius) & base
    candidate = base & ~protected & (score >= float(threshold))

    labels, n = ndi.label(candidate)
    regions = []
    for lab in range(1, n+1):
        reg = labels == lab
        regions.append((int(reg.sum()), lab, reg))
    regions.sort()

    image_budget = max(1, int(np.floor(cfg.max_total_remove_fraction * base.size)))
    fg_budget = max(1, int(np.floor(cfg.max_foreground_remove_fraction * int(base.sum()))))
    total_budget = min(image_budget, fg_budget)
    component_cap = max(1, int(np.floor(cfg.max_component_fraction * base.size)))

    removed = np.zeros_like(base)
    used = 0
    base_components = int(ndi.label(base)[1])
    for size, _, reg in regions:
        if size > component_cap or used + size > total_budget:
            continue
        trial_removed = removed | reg
        trial = base & ~trial_removed
        if cfg.preserve_component_count and int(ndi.label(trial)[1]) > base_components:
            continue
        removed = trial_removed
        used += size

    refined = base & ~removed
    if np.any(removed & ~base):
        raise AssertionError("safety projection removed outside Base")
    if np.any(removed & protected):
        raise AssertionError("safety projection removed protected core")
    return refined, removed, {
        "status": "PASS",
        "removed_pixels": int(removed.sum()),
        "foreground_pixels": int(base.sum()),
        "protected_pixels": int(protected.sum()),
        "total_budget": int(total_budget),
        "component_cap": int(component_cap),
    }
