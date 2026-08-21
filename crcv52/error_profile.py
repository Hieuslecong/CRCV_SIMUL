from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .counterfactual_errors import CounterfactualConfig


def _component_stats(mask: np.ndarray, reference: np.ndarray) -> list[dict]:
    """Return backward-compatible plus scale-normalized component statistics."""
    mask = np.asarray(mask, bool)
    reference = np.asarray(reference, bool)
    h, w = mask.shape
    diag = max(float(np.hypot(h, w)), 1.0)
    labels, n = ndi.label(mask)

    if reference.any():
        dist_ref, nearest = ndi.distance_transform_edt(~reference, return_indices=True)
        reference_radius = ndi.distance_transform_edt(reference).astype(np.float32)
        nearest_radius = reference_radius[nearest[0], nearest[1]]
    else:
        dist_ref = np.full(mask.shape, diag, dtype=np.float32)
        nearest_radius = np.ones(mask.shape, dtype=np.float32)

    out = []
    for lab in range(1, n + 1):
        reg = labels == lab
        area = int(reg.sum())
        if area == 0:
            continue
        sk_len = int(skeletonize(reg).sum())
        dist = dist_ref[reg]
        width_scale = np.maximum(nearest_radius[reg], 1.0)
        width_normalized_distance = dist / width_scale
        out.append({
            # Legacy absolute fields retained for old evidence readers.
            "area": area,
            "skeleton_length": sk_len,
            "min_distance_to_reference": float(dist.min()),
            "median_distance_to_reference": float(np.median(dist)),
            # V5.20.2 scale-aware fields.
            "image_diagonal": diag,
            "area_fraction": float(area / mask.size),
            "skeleton_length_norm": float(sk_len / diag),
            "min_distance_to_reference_norm": float(np.min(width_normalized_distance)),
            "median_distance_to_reference_norm": float(np.median(width_normalized_distance)),
            "median_reference_radius": float(np.median(width_scale)),
        })
    return out


def profile_base_error(base_mask, gt_mask) -> dict:
    """Profile natural Base errors. Caller must restrict this to FIT."""
    base = np.asarray(base_mask, bool)
    gt = np.asarray(gt_mask, bool)
    if base.shape != gt.shape:
        raise ValueError("base_mask and gt_mask must match")
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
    return {
        "n_images": len(profiles),
        "image_diagonal": float(np.median(diagonals)) if diagonals else 1.0,
        "fp_components": [c for p in profiles for c in p.get("fp_components", [])],
        "fn_components": [c for p in profiles for c in p.get("fn_components", [])],
        "fp_pixels": int(sum(p.get("fp_pixels", 0) for p in profiles)),
        "fn_pixels": int(sum(p.get("fn_pixels", 0) for p in profiles)),
        "gt_pixels": int(sum(p.get("gt_pixels", 0) for p in profiles)),
    }


def _median(values, default):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def calibrate_counterfactual_config(profile: dict) -> CounterfactualConfig:
    """Derive simulator scales from FIT-only, normalized natural-error statistics.

    Attached/detached grouping is based on distance relative to local reference
    width, not an absolute pixel threshold. Integer simulator radii are recovered
    at the current profile resolution only at the final step.
    """
    fps = profile.get("fp_components", [])
    fns = profile.get("fn_components", [])
    diag = max(float(profile.get("image_diagonal", 1.0)), 1.0)

    attached = [c for c in fps if c.get("min_distance_to_reference_norm", float("inf")) <= 1.5]
    detached = [c for c in fps if c.get("min_distance_to_reference_norm", float("inf")) > 1.5]

    spur_norm = _median([c.get("skeleton_length_norm", np.nan) for c in attached], 0.033)
    spur_len = int(round(spur_norm * diag))
    spur_len = int(np.clip(spur_len, max(2, round(0.015*diag)), max(3, round(0.09*diag))))

    blob_area_fraction = _median([c.get("area_fraction", np.nan) for c in detached], math.pi*(0.011**2))
    blob_radius = int(round(math.sqrt(max(blob_area_fraction, 1e-8) * (diag*diag/2.0) / math.pi)))
    blob_radius = int(np.clip(blob_radius, max(1, round(0.004*diag)), max(2, round(0.03*diag))))

    shell_ratio = _median([c.get("median_distance_to_reference_norm", np.nan) for c in attached], 1.0)
    local_radius = _median([c.get("median_reference_radius", np.nan) for c in attached], max(1.0, 0.006*diag))
    dilation_radius = int(round(shell_ratio * local_radius))
    dilation_radius = int(np.clip(dilation_radius, 1, max(1, round(0.02*diag))))

    # FN area is normalized by GT foreground mass to remain comparable across
    # resolutions and dataset crops.
    gt_pixels = max(int(profile.get("gt_pixels", 0)), 1)
    fn_area = _median([c["area"] for c in fns], max(1.0, 0.006 * gt_pixels))
    gap_fraction = float(np.clip(np.sqrt(max(fn_area, 1.0)) / np.sqrt(gt_pixels), 0.025, 0.12))
    endpoint_fraction = float(np.clip(0.7 * gap_fraction, 0.02, 0.10))

    return CounterfactualConfig(
        gap_fraction=gap_fraction,
        endpoint_fraction=endpoint_fraction,
        dilation_radius=dilation_radius,
        spur_length=spur_len,
        spur_radius=max(1, round(0.006*diag)),
        blob_radius=blob_radius,
        bridge_radius=max(1, round(0.006*diag)),
    )
