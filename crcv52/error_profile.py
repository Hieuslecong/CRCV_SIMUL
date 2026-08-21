from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .counterfactual_errors import CounterfactualConfig
from .region_supervision import distance_ratio_to_reference


def _shape_stats(reg: np.ndarray, diag: float) -> tuple[float, float, float]:
    ys, xs = np.where(reg)
    if len(ys) == 0:
        return 0.0, 1.0, 0.0
    sk_len_norm = float(skeletonize(reg).sum()) / diag
    if len(ys) < 3:
        return sk_len_norm, 1.0, 0.0
    pts = np.stack([ys, xs], axis=1).astype(np.float32)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    elong = float(np.sqrt(vals[-1] / vals[0]))
    v = vecs[:, -1]
    angle = float(np.arctan2(v[0], v[1]))
    return sk_len_norm, elong, angle


def _component_stats(mask: np.ndarray, reference: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, bool)
    reference = np.asarray(reference, bool)
    if mask.ndim != 2 or reference.ndim != 2 or mask.shape != reference.shape:
        raise ValueError("mask/reference must be same-shape 2-D masks")
    h, w = mask.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    labels, n = ndi.label(mask)

    if reference.any():
        dist_ref, nearest = ndi.distance_transform_edt(~reference, return_indices=True)
        reference_radius = ndi.distance_transform_edt(reference).astype(np.float32)
        nearest_radius = np.maximum(reference_radius[nearest[0], nearest[1]], 1.0)
        dist_ratio = distance_ratio_to_reference(reference)
    else:
        dist_ref = np.full(mask.shape, diag, dtype=np.float32)
        nearest_radius = np.ones(mask.shape, dtype=np.float32)
        dist_ratio = np.full(mask.shape, np.inf, dtype=np.float32)

    out = []
    for lab in range(1, n + 1):
        reg = labels == lab
        area = int(reg.sum())
        if area == 0:
            continue
        sk_norm, elong, angle = _shape_stats(reg, diag)
        dist = dist_ref[reg]
        ratio = dist_ratio[reg]
        min_ratio = float(np.min(ratio))
        if min_ratio > 1.5:
            error_type = "detached"
        elif elong >= 2.5 and sk_norm >= 0.01:
            error_type = "attached_elongated"
        else:
            error_type = "attached_shell_or_compact"
        out.append({
            "area": area,
            "skeleton_length": int(round(sk_norm * diag)),
            "min_distance_to_reference": float(dist.min()),
            "median_distance_to_reference": float(np.median(dist)),
            "image_diagonal": diag,
            "area_fraction": float(area / mask.size),
            "skeleton_length_norm": sk_norm,
            "elongation": elong,
            "orientation_rad": angle,
            "min_distance_to_reference_norm": min_ratio,
            "median_distance_to_reference_norm": float(np.median(ratio)),
            "median_reference_radius": float(np.median(nearest_radius[reg])),
            "error_type": error_type,
        })
    return out


def profile_base_error(base_mask, gt_mask) -> dict:
    base = np.asarray(base_mask, bool)
    gt = np.asarray(gt_mask, bool)
    if base.ndim != 2 or gt.ndim != 2 or base.shape != gt.shape:
        raise ValueError("base_mask and gt_mask must be same-shape 2-D masks")
    fp = base & ~gt
    fn = gt & ~base
    h, w = base.shape
    return {
        "shape": [int(h), int(w)],
        "image_diagonal": float(np.hypot(h, w)),
        "gt_pixels": int(gt.sum()),
        "base_pixels": int(base.sum()),
        "fp_pixels": int(fp.sum()),
        "fn_pixels": int(fn.sum()),
        "fp_components": _component_stats(fp, gt),
        "fn_components": _component_stats(fn, base),
    }


def merge_error_profiles(profiles: list[dict]) -> dict:
    diagonals = [float(p.get("image_diagonal", 0.0)) for p in profiles if p.get("image_diagonal")]
    fp = [c for p in profiles for c in p.get("fp_components", [])]
    fn = [c for p in profiles for c in p.get("fn_components", [])]
    by_type = {}
    for c in fp:
        by_type.setdefault(c.get("error_type", "unknown"), []).append(c)
    return {
        "n_images": len(profiles),
        "image_diagonal": float(np.median(diagonals)) if diagonals else 1.0,
        "fp_components": fp,
        "fn_components": fn,
        "fp_components_by_type": by_type,
        "fp_pixels": int(sum(p.get("fp_pixels", 0) for p in profiles)),
        "fn_pixels": int(sum(p.get("fn_pixels", 0) for p in profiles)),
        "gt_pixels": int(sum(p.get("gt_pixels", 0) for p in profiles)),
    }


def _median(values, default):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def calibrate_counterfactual_config(profile: dict) -> CounterfactualConfig:
    """Derive simulator scales from FIT-only, error-type-aware statistics."""
    fps = profile.get("fp_components", [])
    fns = profile.get("fn_components", [])
    diag = max(float(profile.get("image_diagonal", 1.0)), 1.0)
    elongated = [c for c in fps if c.get("error_type") == "attached_elongated"]
    shells = [c for c in fps if c.get("error_type") == "attached_shell_or_compact"]
    detached = [c for c in fps if c.get("error_type") == "detached"]

    spur_norm = _median([c.get("skeleton_length_norm", np.nan) for c in elongated], 0.033)
    spur_len = int(np.clip(round(spur_norm * diag),
                           max(2, round(0.015*diag)), max(3, round(0.09*diag))))

    blob_area_fraction = _median([c.get("area_fraction", np.nan) for c in detached],
                                 math.pi*(0.011**2))
    blob_radius = int(round(math.sqrt(max(blob_area_fraction, 1e-8) *
                                      (diag*diag/2.0) / math.pi)))
    blob_radius = int(np.clip(blob_radius,
                              max(1, round(0.004*diag)), max(2, round(0.03*diag))))

    shell_ratio = _median([c.get("median_distance_to_reference_norm", np.nan) for c in shells], 1.0)
    local_radius = _median([c.get("median_reference_radius", np.nan) for c in shells],
                           max(1.0, 0.006*diag))
    dilation_radius = int(np.clip(round(shell_ratio * local_radius),
                                  1, max(1, round(0.02*diag))))

    gt_pixels = max(int(profile.get("gt_pixels", 0)), 1)
    fn_area_fraction_of_gt = _median([c.get("area", 0)/gt_pixels for c in fns], 0.006)
    gap_fraction = float(np.clip(np.sqrt(max(fn_area_fraction_of_gt, 1e-8)), 0.025, 0.12))
    endpoint_fraction = float(np.clip(0.7 * gap_fraction, 0.02, 0.10))

    primitive_radius = max(1, round(0.006*diag))
    return CounterfactualConfig(
        gap_fraction=gap_fraction,
        endpoint_fraction=endpoint_fraction,
        dilation_radius=dilation_radius,
        spur_length=spur_len,
        spur_radius=primitive_radius,
        blob_radius=blob_radius,
        bridge_radius=primitive_radius,
    )
