from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .counterfactual_errors import CounterfactualConfig


def _component_stats(mask: np.ndarray, reference: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, bool)
    reference = np.asarray(reference, bool)
    labels, n = ndi.label(mask)
    dist_ref = ndi.distance_transform_edt(~reference) if reference.any() else np.full(mask.shape, max(mask.shape), float)
    out = []
    for lab in range(1, n + 1):
        reg = labels == lab
        area = int(reg.sum())
        if area == 0:
            continue
        out.append({
            "area": area,
            "skeleton_length": int(skeletonize(reg).sum()),
            "min_distance_to_reference": float(dist_ref[reg].min()),
            "median_distance_to_reference": float(np.median(dist_ref[reg])),
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
    return {
        "gt_pixels": int(gt.sum()),
        "base_pixels": int(base.sum()),
        "fp_pixels": int(fp.sum()),
        "fn_pixels": int(fn.sum()),
        "fp_components": _component_stats(fp, gt),
        "fn_components": _component_stats(fn, base),
    }


def merge_error_profiles(profiles: list[dict]) -> dict:
    return {
        "n_images": len(profiles),
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
    """Derive conservative simulator scales from FIT-only natural Base errors."""
    fps = profile.get("fp_components", [])
    fns = profile.get("fn_components", [])
    attached = [c for c in fps if c["min_distance_to_reference"] <= 1.5]
    detached = [c for c in fps if c["min_distance_to_reference"] > 1.5]

    spur_len = int(np.clip(round(_median([c["skeleton_length"] for c in attached], 6)), 3, 16))
    blob_area = _median([c["area"] for c in detached], math.pi * 2**2)
    blob_radius = int(np.clip(round(math.sqrt(max(blob_area, 1.0) / math.pi)), 1, 5))
    shell_dist = _median([c["median_distance_to_reference"] for c in attached], 1.0)
    dilation_radius = int(np.clip(round(shell_dist), 1, 3))

    gt_pixels = max(int(profile.get("gt_pixels", 0)), 1)
    fn_area = _median([c["area"] for c in fns], max(1.0, 0.006 * gt_pixels))
    gap_fraction = float(np.clip(np.sqrt(max(fn_area, 1.0)) / np.sqrt(gt_pixels), 0.025, 0.12))
    endpoint_fraction = float(np.clip(0.7 * gap_fraction, 0.02, 0.10))

    return CounterfactualConfig(
        gap_fraction=gap_fraction,
        endpoint_fraction=endpoint_fraction,
        dilation_radius=dilation_radius,
        spur_length=spur_len,
        spur_radius=1,
        blob_radius=blob_radius,
        bridge_radius=1,
    )
