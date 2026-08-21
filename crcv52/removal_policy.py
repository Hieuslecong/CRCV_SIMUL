from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .region_supervision import KEEP, REMOVE, RemoveSupervisionConfig, build_remove_supervision, topology_hard_keep_mask


@dataclass(frozen=True)
class RemovalFeatureConfig:
    # Legacy absolute override. None means use the scale-aware normalized sigma.
    blur_sigma: float | None = None
    blur_sigma_norm: float = 0.006


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a.astype(np.float32)
    lo = float(np.quantile(a, 0.02)); hi = float(np.quantile(a, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a-lo)/(hi-lo+1e-6), 0, 1).astype(np.float32)


def _skeleton_context(base: np.ndarray):
    sk = skeletonize(base)
    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    if not sk.any():
        far = np.ones(base.shape, np.float32)
        zeros = np.zeros(base.shape, np.float32)
        return sk, far, far, far, zeros, zeros

    dist_sk, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    inside_radius = ndi.distance_transform_edt(base).astype(np.float32)
    nearest_radius = inside_radius[inds[0], inds[1]]
    radial_ratio = np.clip(dist_sk / (nearest_radius + 1e-3), 0, 4).astype(np.float32)

    neigh = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant")
    endpoints = sk & (neigh <= 2)
    junctions = sk & (neigh >= 4)
    dist_endpoint = (ndi.distance_transform_edt(~endpoints) / diag).astype(np.float32) if endpoints.any() else np.ones(base.shape, np.float32)
    dist_junction = (ndi.distance_transform_edt(~junctions) / diag).astype(np.float32) if junctions.any() else np.ones(base.shape, np.float32)
    nearest_degree = neigh[inds[0], inds[1]].astype(np.float32) / 9.0
    return sk, (dist_sk/diag).astype(np.float32), dist_endpoint, dist_junction, radial_ratio, nearest_degree


def _component_maps(base: np.ndarray):
    labels, n = ndi.label(base)
    area_frac = np.zeros(base.shape, np.float32)
    sklen_frac = np.zeros(base.shape, np.float32)
    elongation = np.zeros(base.shape, np.float32)
    bbox_diag_frac = np.zeros(base.shape, np.float32)
    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    for lab in range(1, n+1):
        reg = labels == lab
        ys, xs = np.where(reg)
        if len(ys) == 0:
            continue
        area_frac[reg] = len(ys) / float(base.size)
        sklen_frac[reg] = float(skeletonize(reg).sum()) / diag
        bh = int(ys.max()-ys.min()+1); bw = int(xs.max()-xs.min()+1)
        bbox_diag_frac[reg] = float(np.hypot(bh, bw) / diag)
        if len(ys) >= 3:
            pts = np.stack([ys, xs], axis=1).astype(np.float32)
            cov = np.cov(pts, rowvar=False)
            vals = np.maximum(np.linalg.eigvalsh(cov), 1e-6)
            elong = float(np.sqrt(vals[-1] / vals[0]))
        else:
            elong = 1.0
        elongation[reg] = min(elong, 20.0) / 20.0
    return area_frac, sklen_frac, elongation, bbox_diag_frac


def build_removal_features(image, probability, base_mask,
                           config: RemovalFeatureConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Build GT-free appearance + topology-aware features for KEEP/REMOVE."""
    cfg = config or RemovalFeatureConfig()
    image = np.asarray(image, dtype=np.float32)
    prob = np.asarray(probability, dtype=np.float32)
    base = np.asarray(base_mask, dtype=bool)
    if image.ndim != 3 or image.shape[:2] != base.shape or prob.shape != base.shape:
        raise ValueError("bad image/probability/base shapes")
    if image.shape[2] < 1:
        raise ValueError("image must have at least one channel")
    if not np.isfinite(image).all() or not np.isfinite(prob).all():
        raise ValueError("image/probability contains non-finite values")
    h, w = base.shape
    diag = max(float(np.hypot(h,w)), 1.0)
    sigma = float(cfg.blur_sigma) if cfg.blur_sigma is not None else max(0.5, float(cfg.blur_sigma_norm) * diag)

    gray = image.mean(axis=2)
    blur = ndi.gaussian_filter(gray, sigma)
    pblur = ndi.gaussian_filter(prob, sigma)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx*gx + gy*gy)
    inside = ndi.distance_transform_edt(base).astype(np.float32)
    _, dist_sk, dist_ep, dist_jn, radial_ratio, nearest_degree = _skeleton_context(base)
    comp_area, comp_sklen, comp_elong, comp_bbox_diag = _component_maps(base)
    local_radius = np.clip(inside/diag, 0, 1)

    X = np.stack([
        prob, pblur, gray, gray-blur, _norm(grad),
        dist_sk, local_radius, radial_ratio, dist_ep, dist_jn, nearest_degree,
        comp_area, comp_sklen, comp_elong, comp_bbox_diag,
    ], axis=-1).astype(np.float32)
    names = [
        "prob", "blur_prob", "gray", "gray_minus_blur", "gray_gradient",
        "distance_to_base_skeleton_norm", "inside_radius_norm", "radial_position_ratio",
        "distance_to_endpoint_norm", "distance_to_junction_norm", "nearest_skeleton_degree",
        "component_area_fraction", "component_skeleton_length_norm", "component_elongation",
        "component_bbox_diagonal_norm",
    ]
    return X, names


def _sample_indices(idx: np.ndarray, n: int, rng) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if n <= 0 or len(idx) == 0:
        return np.empty(0, np.int64)
    if len(idx) <= n:
        return idx
    return np.asarray(rng.choice(idx, n, replace=False), dtype=np.int64)


def sample_keep_remove_training(image, probability, base_mask, gt_mask,
                                max_keep: int = 1400,
                                max_remove: int = 450,
                                max_per_class: int | None = None,
                                boundary_keep_fraction: float = 0.55,
                                hard_keep_radius_scale: float = 1.0,
                                remove_min_distance_ratio: float = 1.25,
                                supervision_mode: str = "detached_component",
                                hard_keep_repeat: int = 3,
                                remove_exclusion_radius: int | None = None,
                                require_detached_remove: bool | None = None,
                                seed: int = 1337) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Scale-aware KEEP-heavy sampling with explicit IGNORE supervision.

    Canonical V5.20.3 uses ``detached_component`` supervision. The legacy keyword
    arguments ``remove_exclusion_radius`` and ``require_detached_remove`` are
    accepted for backward compatibility only; canonical training does not depend
    on an absolute-pixel exclusion radius.
    """
    if max_per_class is not None:
        max_keep = int(max_per_class)
        max_remove = int(max_per_class)
    if max_keep < 0 or max_remove < 0:
        raise ValueError("sample limits must be non-negative")
    if not 0.0 <= float(boundary_keep_fraction) <= 1.0:
        raise ValueError("boundary_keep_fraction must be in [0,1]")
    if remove_exclusion_radius is not None and int(remove_exclusion_radius) < 0:
        raise ValueError("remove_exclusion_radius must be non-negative or None")
    if require_detached_remove is False:
        supervision_mode = "distal_pixel"
    elif require_detached_remove is True:
        supervision_mode = "detached_component"

    Xmap, names = build_removal_features(image, probability, base_mask)
    base = np.asarray(base_mask, bool); gt = np.asarray(gt_mask, bool)
    supervision = build_remove_supervision(
        base, gt, RemoveSupervisionConfig(
            mode=supervision_mode,
            remove_min_distance_ratio=float(remove_min_distance_ratio),
        )
    )
    keep = supervision == KEEP
    remove = supervision == REMOVE
    rng = np.random.default_rng(seed)

    hard_keep = topology_hard_keep_mask(base, gt, radius_scale=float(hard_keep_radius_scale))
    hard_keep &= keep
    easy_keep = keep & ~hard_keep

    n_hard = int(round(max_keep * float(boundary_keep_fraction)))
    hard_idx = _sample_indices(np.flatnonzero(hard_keep.ravel()), n_hard, rng)
    remaining = max_keep - len(hard_idx)
    easy_idx = _sample_indices(np.flatnonzero(easy_keep.ravel()), remaining, rng)
    chosen = np.concatenate([hard_idx, easy_idx])
    if len(chosen) < max_keep:
        all_keep = np.flatnonzero(keep.ravel())
        used = set(chosen.tolist())
        rest = np.asarray([i for i in all_keep if i not in used], dtype=np.int64)
        fill = _sample_indices(rest, max_keep-len(chosen), rng)
        keep_idx = np.concatenate([chosen, fill])
    else:
        keep_idx = chosen

    rem_idx = _sample_indices(np.flatnonzero(remove.ravel()), max_remove, rng)
    repeat = max(1, int(hard_keep_repeat))
    hard_weighted = np.tile(hard_idx, repeat) if len(hard_idx) else hard_idx
    hard_set = set(hard_idx.tolist())
    keep_nonhard = np.asarray([i for i in keep_idx if i not in hard_set], dtype=np.int64)
    idx = np.concatenate([hard_weighted, keep_nonhard, rem_idx])
    y = np.concatenate([
        np.zeros(len(hard_weighted) + len(keep_nonhard), np.int8),
        np.ones(len(rem_idx), np.int8),
    ])
    if len(idx) == 0:
        return np.empty((0, len(names)), np.float32), y, names
    return Xmap.reshape(-1, Xmap.shape[-1])[idx], y, names
