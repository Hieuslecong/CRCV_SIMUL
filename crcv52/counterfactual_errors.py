from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class CounterfactualConfig:
    gap_fraction: float = 0.08
    endpoint_fraction: float = 0.05
    dilation_radius: int = 1
    spur_length: int = 6
    spur_radius: int = 1
    blob_radius: int = 2
    bridge_radius: int = 1


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx * xx + yy * yy <= r * r)


def _dilate_primitive(primitive: np.ndarray, radius: int) -> np.ndarray:
    """Dilate only the synthetic primitive, never the complete GT crack."""
    if radius <= 0:
        return primitive.astype(bool)
    return ndi.binary_dilation(primitive, structure=_disk(radius))


def simulate_corruption(gt_mask, operator: str, seed: int = 1337,
                        config: CounterfactualConfig | None = None) -> tuple[np.ndarray, dict]:
    """Create a controlled corrupted Base-like mask from a clean crack mask.

    Additive operators thicken only the newly-created primitive. They never dilate
    the complete GT crack, preventing false REMOVE labels on the true boundary.
    Production training should calibrate the configuration from FIT-only Base-error
    profiles rather than treating these dataclass defaults as scientific constants.
    """
    cfg = config or CounterfactualConfig()
    gt = np.asarray(gt_mask, dtype=bool)
    if gt.ndim != 2:
        raise ValueError("gt_mask must be 2-D")
    if min(gt.shape) < 2:
        raise ValueError("gt_mask is too small")
    rng = np.random.default_rng(seed)
    out = gt.copy()
    sk = skeletonize(gt)
    ys, xs = np.where(sk)
    meta = {"operator": operator, "seed": int(seed), "status": "APPLIED"}

    if operator == "boundary_erode":
        out = ndi.binary_erosion(gt, structure=_disk(1), border_value=0)
    elif operator == "width_dilate":
        out = ndi.binary_dilation(gt, structure=_disk(cfg.dilation_radius))
    elif operator == "gap_delete":
        if len(ys) == 0:
            return gt.copy(), {**meta, "status": "NO_OP_EMPTY"}
        i = int(rng.integers(0, len(ys)))
        radius = max(1, int(round(np.sqrt(max(gt.sum(), 1)) * cfg.gap_fraction)))
        yy, xx = np.ogrid[:gt.shape[0], :gt.shape[1]]
        out[(yy-ys[i])**2 + (xx-xs[i])**2 <= radius**2] = False
        meta["radius"] = radius
    elif operator == "endpoint_truncate":
        if len(ys) == 0:
            return gt.copy(), {**meta, "status": "NO_OP_EMPTY"}
        neigh = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant")
        ey, ex = np.where(sk & (neigh <= 2))
        if len(ey) == 0:
            ey, ex = ys, xs
        i = int(rng.integers(0, len(ey)))
        radius = max(1, int(round(np.sqrt(max(gt.sum(), 1)) * cfg.endpoint_fraction)))
        yy, xx = np.ogrid[:gt.shape[0], :gt.shape[1]]
        out[(yy-ey[i])**2 + (xx-ex[i])**2 <= radius**2] = False
        meta["radius"] = radius
    elif operator == "side_spur":
        if len(ys) == 0:
            return gt.copy(), {**meta, "status": "NO_OP_EMPTY"}
        i = int(rng.integers(0, len(ys)))
        y0, x0 = int(ys[i]), int(xs[i])
        angle = float(rng.uniform(0, 2*np.pi))
        primitive = np.zeros_like(gt)
        for k in range(1, max(1, int(cfg.spur_length)) + 1):
            y = int(round(y0 + k*np.sin(angle)))
            x = int(round(x0 + k*np.cos(angle)))
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                primitive[y, x] = True
        primitive = _dilate_primitive(primitive, cfg.spur_radius)
        out = gt | primitive
        meta.update({"spur_length": int(cfg.spur_length), "spur_radius": int(cfg.spur_radius)})
    elif operator == "isolated_blob":
        radius = max(1, int(cfg.blob_radius))
        # Keep sampling valid even for very small unit/smoke-test masks.
        ylo = min(radius, gt.shape[0]-1)
        xlo = min(radius, gt.shape[1]-1)
        yhi = max(ylo+1, gt.shape[0]-radius)
        xhi = max(xlo+1, gt.shape[1]-radius)
        y = int(rng.integers(ylo, yhi))
        x = int(rng.integers(xlo, xhi))
        yy, xx = np.ogrid[:gt.shape[0], :gt.shape[1]]
        primitive = ((yy-y)**2 + (xx-x)**2 <= radius**2)
        out = gt | primitive
        meta["blob_radius"] = radius
    elif operator == "false_bridge":
        labels, n = ndi.label(gt)
        if n < 2:
            return gt.copy(), {**meta, "status": "NO_OP_NEEDS_COMPONENTS"}
        centers = ndi.center_of_mass(gt, labels, range(1, n+1))
        best = None
        for a in range(n):
            for b in range(a+1, n):
                ya, xa = centers[a]; yb, xb = centers[b]
                d2 = (ya-yb)**2 + (xa-xb)**2
                if best is None or d2 < best[0]:
                    best = (d2, a, b)
        _, ia, ib = best
        (y0,x0),(y1,x1) = centers[ia], centers[ib]
        steps = max(abs(int(y1-y0)), abs(int(x1-x0)), 2)
        primitive = np.zeros_like(gt)
        for a in np.linspace(0,1,steps):
            y = int(round((1-a)*y0+a*y1)); x = int(round((1-a)*x0+a*x1))
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                primitive[y,x] = True
        primitive = _dilate_primitive(primitive, cfg.bridge_radius)
        out = gt | primitive
        meta["bridge_radius"] = int(cfg.bridge_radius)
    else:
        raise ValueError(f"unknown operator: {operator}")

    return out.astype(bool), meta
