from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class RemovalFeatureConfig:
    blur_sigma: float = 1.0


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo = float(np.quantile(a, 0.02)); hi = float(np.quantile(a, 0.98))
    return np.clip((a-lo)/(hi-lo+1e-6), 0, 1).astype(np.float32)


def _skeleton_context(base: np.ndarray):
    sk = skeletonize(base)
    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    if not sk.any():
        far = np.full(base.shape, 1.0, np.float32)
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
    h, w = base.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    for lab in range(1, n+1):
        reg = labels == lab
        ys, xs = np.where(reg)
        if len(ys) == 0:
            continue
        area_frac[reg] = len(ys) / float(base.size)
        sklen_frac[reg] = float(skeletonize(reg).sum()) / diag
        if len(ys) >= 3:
            pts = np.stack([ys, xs], axis=1).astype(np.float32)
            cov = np.cov(pts, rowvar=False)
            vals = np.maximum(np.linalg.eigvalsh(cov), 1e-6)
            elong = float(np.sqrt(vals[-1] / vals[0]))
        else:
            elong = 1.0
        elongation[reg] = min(elong, 20.0) / 20.0
    return area_frac, sklen_frac, elongation


def build_removal_features(image, probability, base_mask,
                           config: RemovalFeatureConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Build GT-free appearance + topology-aware features for KEEP/REMOVE.

    GT is intentionally absent from this runtime feature API. Skeleton, width,
    endpoint/junction and component descriptors are derived only from frozen Base.
    Absolute x/y coordinates are deliberately excluded to reduce dataset-position bias.
    """
    cfg = config or RemovalFeatureConfig()
    image = np.asarray(image, dtype=np.float32)
    prob = np.asarray(probability, dtype=np.float32)
    base = np.asarray(base_mask, dtype=bool)
    if image.ndim != 3 or image.shape[:2] != base.shape or prob.shape != base.shape:
        raise ValueError("bad image/probability/base shapes")
    gray = image.mean(axis=2)
    blur = ndi.gaussian_filter(gray, cfg.blur_sigma)
    pblur = ndi.gaussian_filter(prob, cfg.blur_sigma)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx*gx + gy*gy)
    inside = ndi.distance_transform_edt(base).astype(np.float32)
    h, w = base.shape
    diag = max(float(np.hypot(h,w)), 1.0)
    _, dist_sk, dist_ep, dist_jn, radial_ratio, nearest_degree = _skeleton_context(base)
    comp_area, comp_sklen, comp_elong = _component_maps(base)
    local_radius = np.clip(inside/diag, 0, 1)

    X = np.stack([
        prob, pblur, gray, gray-blur, _norm(grad),
        dist_sk, local_radius, radial_ratio, dist_ep, dist_jn, nearest_degree,
        comp_area, comp_sklen, comp_elong,
    ], axis=-1).astype(np.float32)
    names = [
        "prob", "blur_prob", "gray", "gray_minus_blur", "gray_gradient",
        "distance_to_base_skeleton_norm", "inside_radius_norm", "radial_position_ratio",
        "distance_to_endpoint_norm", "distance_to_junction_norm", "nearest_skeleton_degree",
        "component_area_fraction", "component_skeleton_length_norm", "component_elongation",
    ]
    return X, names


def _sample_indices(idx: np.ndarray, n: int, rng) -> np.ndarray:
    if len(idx) <= n:
        return idx
    return rng.choice(idx, n, replace=False)


def sample_keep_remove_training(image, probability, base_mask, gt_mask,
                                max_keep: int = 1400,
                                max_remove: int = 450,
                                max_per_class: int | None = None,
                                boundary_keep_fraction: float = 0.55,
                                hard_keep_radius: int = 2,
                                remove_exclusion_radius: int = 1,
                                hard_keep_repeat: int = 3,
                                require_detached_remove: bool = True,
                                seed: int = 1337) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample KEEP/REMOVE with explicit boundary/topology KEEP emphasis.

    REMOVE is deliberately not class-balanced. KEEP dominates. Hard KEEP examples
    come from true crack boundary and endpoint/junction neighborhoods. Pixels in a
    small GT tolerance band are excluded from REMOVE supervision so annotation-edge
    uncertainty is not taught as a deletion target.
    """
    from .action_targets import build_action_targets
    if max_per_class is not None:
        max_keep = int(max_per_class)
        max_remove = int(max_per_class)
    Xmap, names = build_removal_features(image, probability, base_mask)
    t = build_action_targets(base_mask, gt_mask)
    base = np.asarray(base_mask, bool); gt = np.asarray(gt_mask, bool)
    keep = t["keep"]
    rng = np.random.default_rng(seed)

    eroded = ndi.binary_erosion(gt, structure=np.ones((3,3), bool), border_value=0)
    boundary = gt & ~eroded
    sk = skeletonize(gt)
    neigh = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant") if sk.any() else np.zeros_like(gt, np.uint8)
    topo = sk & ((neigh <= 2) | (neigh >= 4))
    if hard_keep_radius > 0 and topo.any():
        topo = ndi.binary_dilation(topo, iterations=hard_keep_radius)
    hard_keep = keep & (boundary | topo)
    easy_keep = keep & ~hard_keep

    n_hard = int(round(max_keep * boundary_keep_fraction))
    hard_idx = _sample_indices(np.flatnonzero(hard_keep.ravel()), n_hard, rng)
    remaining = max_keep - len(hard_idx)
    easy_idx = _sample_indices(np.flatnonzero(easy_keep.ravel()), remaining, rng)
    if len(hard_idx) + len(easy_idx) < max_keep:
        all_keep = np.flatnonzero(keep.ravel())
        chosen = set(np.concatenate([hard_idx,easy_idx]).tolist())
        rest = np.array([i for i in all_keep if i not in chosen], dtype=np.int64)
        fill = _sample_indices(rest, max_keep-len(chosen), rng)
        keep_idx = np.concatenate([hard_idx,easy_idx,fill])
    else:
        keep_idx = np.concatenate([hard_idx,easy_idx])

    if remove_exclusion_radius > 0:
        tol_gt = ndi.binary_dilation(gt, iterations=remove_exclusion_radius)
    else:
        tol_gt = gt
    fp = base & ~gt
    if require_detached_remove:
        flab, fn = ndi.label(fp)
        safe_remove = np.zeros_like(base)
        for lab in range(1, fn + 1):
            reg = flab == lab
            if not np.any(ndi.binary_dilation(reg, iterations=1) & tol_gt):
                safe_remove |= reg
    else:
        safe_remove = fp & ~tol_gt
    rem_idx = _sample_indices(np.flatnonzero(safe_remove.ravel()), max_remove, rng)

    repeat = max(1, int(hard_keep_repeat))
    hard_weighted = np.tile(hard_idx, repeat) if len(hard_idx) else hard_idx
    idx = np.concatenate([hard_weighted, easy_idx, rem_idx])
    y = np.concatenate([
        np.zeros(len(hard_weighted) + len(easy_idx), np.int8),
        np.ones(len(rem_idx), np.int8),
    ])
    if len(idx) == 0:
        return np.empty((0, len(names)), np.float32), y, names
    return Xmap.reshape(-1, Xmap.shape[-1])[idx], y, names
