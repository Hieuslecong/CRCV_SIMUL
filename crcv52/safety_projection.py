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
    protect_parallel_boundary_strips: bool = True
    boundary_strip_min_skeleton_length: int = 6
    boundary_strip_max_median_distance: float = 2.5


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    disk = (xx*xx + yy*yy) <= r*r
    return ndi.binary_dilation(mask, structure=disk)


def _looks_like_parallel_boundary_strip(reg: np.ndarray, base_skeleton: np.ndarray,
                                        cfg: SafetyProjectionConfig) -> bool:
    """Reject long REMOVE regions that run immediately beside the Base skeleton.

    Such regions are the characteristic failure mode of pixel-wise suppression:
    a true crack boundary is shaved off as a long shell. A short spur/blob may be
    close to the skeleton at its attachment point, but its median distance is
    typically larger and/or its own skeleton is short.
    """
    if not cfg.protect_parallel_boundary_strips or not reg.any() or not base_skeleton.any():
        return False
    reg_sk = skeletonize(reg)
    sk_len = int(reg_sk.sum())
    if sk_len < cfg.boundary_strip_min_skeleton_length:
        return False
    dist_to_base_sk = ndi.distance_transform_edt(~base_skeleton)
    med = float(np.median(dist_to_base_sk[reg]))
    return med <= cfg.boundary_strip_max_median_distance


def project_remove(base_mask, remove_score, threshold: float,
                   config: SafetyProjectionConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project REMOVE scores into a conservative, GT-free corrected mask.

    Runtime inputs are only Base and REMOVE score. The projection protects the
    Base skeleton/core and rejects long high-score strips that run parallel to
    that skeleton, preventing systematic crack-boundary erosion. GT is neither
    accepted nor required at runtime.
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
    protected_boundary_regions = 0
    for lab in range(1, n+1):
        reg = labels == lab
        if _looks_like_parallel_boundary_strip(reg, sk, cfg):
            protected_boundary_regions += 1
            continue
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
        "protected_parallel_boundary_regions": int(protected_boundary_regions),
        "total_budget": int(total_budget),
        "component_cap": int(component_cap),
    }
