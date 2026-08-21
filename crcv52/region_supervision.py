from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

IGNORE = np.int8(0)
KEEP = np.int8(1)
REMOVE = np.int8(2)


@dataclass(frozen=True)
class RemoveSupervisionConfig:
    """Scale-aware training supervision inside the frozen Base mask.

    `detached_component` is the safety-first default: an FP component is REMOVE
    only when the complete component is sufficiently separated from GT in units
    of local GT half-width. Attached structures are IGNORE rather than being
    mislabeled as crack boundary. `distal_pixel` is retained only for controlled
    ablation and is not the canonical training mode.
    """
    mode: str = "detached_component"
    remove_min_distance_ratio: float = 1.25
    min_reference_radius: float = 1.0


def distance_ratio_to_reference(reference_mask: np.ndarray,
                                min_reference_radius: float = 1.0) -> np.ndarray:
    ref = np.asarray(reference_mask, bool)
    if ref.ndim != 2:
        raise ValueError("reference_mask must be 2-D")
    if not ref.any():
        return np.full(ref.shape, np.inf, np.float32)
    dist = ndi.distance_transform_edt(~ref).astype(np.float32)
    sk = skeletonize(ref)
    if not sk.any():
        return dist / max(float(min_reference_radius), 1.0)
    _, nearest_sk = ndi.distance_transform_edt(~sk, return_indices=True)
    radius = ndi.distance_transform_edt(ref).astype(np.float32)
    nearest_radius = np.maximum(radius[nearest_sk[0], nearest_sk[1]], float(min_reference_radius))
    return (dist / nearest_radius).astype(np.float32)


def build_remove_supervision(base_mask, gt_mask,
                             config: RemoveSupervisionConfig | None = None) -> np.ndarray:
    """Return 0=IGNORE, 1=KEEP, 2=REMOVE supervision inside Base.

    GT is training-only. Runtime code never calls this function.
    """
    cfg = config or RemoveSupervisionConfig()
    if cfg.mode not in {"detached_component", "distal_pixel"}:
        raise ValueError("mode must be 'detached_component' or 'distal_pixel'")
    base = np.asarray(base_mask, bool)
    gt = np.asarray(gt_mask, bool)
    if base.ndim != 2 or gt.ndim != 2 or base.shape != gt.shape:
        raise ValueError("base_mask and gt_mask must be same-shape 2-D masks")
    out = np.full(base.shape, IGNORE, np.int8)
    keep = base & gt
    fp = base & ~gt
    out[keep] = KEEP
    if not fp.any():
        return out
    if not gt.any():
        out[fp] = REMOVE
        return out

    ratio = distance_ratio_to_reference(gt, cfg.min_reference_radius)
    threshold = float(cfg.remove_min_distance_ratio)
    if cfg.mode == "distal_pixel":
        out[fp & (ratio >= threshold)] = REMOVE
        return out

    labels, n = ndi.label(fp)
    for lab in range(1, n + 1):
        reg = labels == lab
        if reg.any() and float(np.min(ratio[reg])) >= threshold:
            out[reg] = REMOVE
    return out


def topology_hard_keep_mask(base_mask, gt_mask, radius_scale: float = 1.0) -> np.ndarray:
    """Scale-aware boundary/endpoint/junction KEEP emphasis mask."""
    base = np.asarray(base_mask, bool)
    gt = np.asarray(gt_mask, bool)
    if base.ndim != 2 or gt.ndim != 2 or base.shape != gt.shape:
        raise ValueError("base_mask and gt_mask must be same-shape 2-D masks")
    keep = base & gt
    if not keep.any():
        return np.zeros_like(base)
    eroded = ndi.binary_erosion(gt, structure=np.ones((3, 3), bool), border_value=0)
    boundary = gt & ~eroded
    sk = skeletonize(gt)
    if not sk.any():
        return keep & boundary
    neigh = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    anchors = sk & ((neigh <= 2) | (neigh >= 4))
    gt_radius = ndi.distance_transform_edt(gt)
    local = float(np.median(gt_radius[sk])) if sk.any() else 1.0
    r = max(1, int(round(max(local, 1.0) * float(radius_scale))))
    anchor_band = ndi.binary_dilation(anchors, iterations=r) if anchors.any() else np.zeros_like(gt)
    return keep & (boundary | anchor_band)
