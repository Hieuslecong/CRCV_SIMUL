from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class SafetyProjectionConfig:
    # Legacy absolute overrides. None means use scale-aware defaults below.
    core_protection_radius: int | None = None
    boundary_strip_min_skeleton_length: int | None = None
    boundary_strip_max_median_distance: float | None = None

    core_protection_local_radius_fraction: float = 0.50
    topology_anchor_local_radius_fraction: float = 1.00
    core_protection_min_radius_norm: float = 0.004
    max_total_remove_fraction: float = 0.03
    max_foreground_remove_fraction: float = 0.10
    max_component_fraction: float = 0.02
    preserve_component_count: bool = True
    protect_parallel_boundary_strips: bool = True
    boundary_strip_min_skeleton_length_norm: float = 0.03
    boundary_strip_max_median_distance_ratio: float = 1.75


def _validate_config(cfg: SafetyProjectionConfig) -> None:
    for name in ("max_total_remove_fraction", "max_foreground_remove_fraction", "max_component_fraction"):
        v = float(getattr(cfg, name))
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if (cfg.core_protection_local_radius_fraction < 0 or
            cfg.topology_anchor_local_radius_fraction < 0 or
            cfg.core_protection_min_radius_norm < 0):
        raise ValueError("core protection scales must be non-negative")
    if cfg.boundary_strip_min_skeleton_length_norm < 0 or cfg.boundary_strip_max_median_distance_ratio < 0:
        raise ValueError("boundary strip scales must be non-negative")


def _skeleton_geometry(base: np.ndarray):
    sk = skeletonize(base)
    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    if not sk.any():
        z = np.zeros(base.shape, np.float32)
        o = np.ones(base.shape, np.float32)
        return sk, z, o, diag
    dist_sk, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    inside_radius = ndi.distance_transform_edt(base).astype(np.float32)
    nearest_radius = np.maximum(inside_radius[inds[0], inds[1]], 1.0)
    return sk, dist_sk.astype(np.float32), nearest_radius.astype(np.float32), diag


def _protected_core(base: np.ndarray, sk: np.ndarray, dist_sk: np.ndarray,
                    nearest_radius: np.ndarray, diag: float,
                    cfg: SafetyProjectionConfig) -> np.ndarray:
    if not sk.any():
        return np.zeros_like(base)
    if cfg.core_protection_radius is not None:
        radius = max(0, int(cfg.core_protection_radius))
        if radius == 0:
            return sk & base
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        disk = (xx*xx + yy*yy) <= radius*radius
        return ndi.binary_dilation(sk, structure=disk) & base
    min_radius = max(1.0, float(cfg.core_protection_min_radius_norm) * diag)
    local_limit = np.maximum(min_radius, float(cfg.core_protection_local_radius_fraction) * nearest_radius)
    protected = base & (dist_sk <= local_limit)

    # Endpoint/junction neighborhoods carry disproportionate topology risk. Protect
    # a full local-radius neighborhood around these anchors rather than only the
    # generic half-width corridor. This prevents one-pixel terminal fragments from
    # escaping the parallel-strip veto after the long strip itself is rejected.
    neigh = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant")
    anchors = sk & ((neigh <= 2) | (neigh >= 4))
    if anchors.any() and cfg.topology_anchor_local_radius_fraction > 0:
        dist_anchor, nearest_anchor = ndi.distance_transform_edt(~anchors, return_indices=True)
        inside_radius = ndi.distance_transform_edt(base).astype(np.float32)
        anchor_radius = np.maximum(inside_radius[nearest_anchor[0], nearest_anchor[1]], min_radius)
        anchor_limit = float(cfg.topology_anchor_local_radius_fraction) * anchor_radius
        protected |= base & (dist_anchor <= anchor_limit)
    return protected


def _looks_like_parallel_boundary_strip(reg: np.ndarray, base_skeleton: np.ndarray,
                                        nearest_radius: np.ndarray, diag: float,
                                        cfg: SafetyProjectionConfig) -> bool:
    if not cfg.protect_parallel_boundary_strips or not reg.any() or not base_skeleton.any():
        return False
    reg_sk = skeletonize(reg)
    sk_len = int(reg_sk.sum())
    min_len = (int(cfg.boundary_strip_min_skeleton_length)
               if cfg.boundary_strip_min_skeleton_length is not None
               else max(2, int(np.ceil(float(cfg.boundary_strip_min_skeleton_length_norm) * diag))))
    if sk_len < min_len:
        return False
    dist_to_base_sk = ndi.distance_transform_edt(~base_skeleton).astype(np.float32)
    if cfg.boundary_strip_max_median_distance is not None:
        return float(np.median(dist_to_base_sk[reg])) <= float(cfg.boundary_strip_max_median_distance)
    ratio = dist_to_base_sk[reg] / np.maximum(nearest_radius[reg], 1.0)
    return float(np.median(ratio)) <= float(cfg.boundary_strip_max_median_distance_ratio)


def project_remove(base_mask, remove_score, threshold: float,
                   config: SafetyProjectionConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project REMOVE scores into a conservative, scale-aware GT-free mask."""
    cfg = config or SafetyProjectionConfig()
    _validate_config(cfg)
    base = np.asarray(base_mask, dtype=bool)
    score = np.asarray(remove_score, dtype=float)
    if base.shape != score.shape or base.ndim != 2:
        raise ValueError("base_mask and remove_score must be same-shape 2-D arrays")
    if not np.isfinite(score).all() or not np.isfinite(float(threshold)):
        raise ValueError("remove_score/threshold contains non-finite values")
    if not base.any():
        return base.copy(), np.zeros_like(base), {"status":"NO_OP_EMPTY_BASE", "removed_pixels":0}

    sk, dist_sk, nearest_radius, diag = _skeleton_geometry(base)
    protected = _protected_core(base, sk, dist_sk, nearest_radius, diag, cfg)
    candidate = base & ~protected & (score >= float(threshold))

    labels, n = ndi.label(candidate)
    regions = []
    protected_boundary_regions = 0
    for lab in range(1, n+1):
        reg = labels == lab
        if _looks_like_parallel_boundary_strip(reg, sk, nearest_radius, diag, cfg):
            protected_boundary_regions += 1
            continue
        regions.append((int(reg.sum()), lab, reg))
    regions.sort(key=lambda x: (x[0], x[1]))

    image_budget = int(np.floor(cfg.max_total_remove_fraction * base.size))
    fg_budget = int(np.floor(cfg.max_foreground_remove_fraction * int(base.sum())))
    total_budget = max(0, min(image_budget, fg_budget))
    component_cap = max(0, int(np.floor(cfg.max_component_fraction * base.size)))

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
    if sk.any() and np.any(sk & ~refined):
        raise AssertionError("safety projection removed Base skeleton")
    return refined, removed, {
        "status": "PASS",
        "removed_pixels": int(removed.sum()),
        "foreground_pixels": int(base.sum()),
        "protected_pixels": int(protected.sum()),
        "protected_parallel_boundary_regions": int(protected_boundary_regions),
        "total_budget": int(total_budget),
        "component_cap": int(component_cap),
        "scale_diagonal": float(diag),
    }
