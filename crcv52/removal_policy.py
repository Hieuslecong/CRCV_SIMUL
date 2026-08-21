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


def build_removal_features(image, probability, base_mask,
                           config: RemovalFeatureConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Build GT-free per-pixel features for KEEP/REMOVE classification.

    Features intentionally combine appearance with Base-derived structure. GT is
    never an input to this function; GT is used only to construct training labels.
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
    sk = skeletonize(base)
    dist_sk = ndi.distance_transform_edt(~sk) if sk.any() else np.full(base.shape, max(base.shape), np.float32)
    dist_bg = ndi.distance_transform_edt(base)
    labels, _ = ndi.label(base)
    sizes = np.bincount(labels.ravel())
    comp_frac = np.zeros(base.shape, np.float32)
    fg = labels > 0
    if fg.any(): comp_frac[fg] = sizes[labels[fg]] / float(base.size)
    yy, xx = np.indices(base.shape)
    h, w = base.shape
    X = np.stack([
        prob,
        pblur,
        gray,
        gray-blur,
        _norm(grad),
        _norm(dist_sk),
        _norm(dist_bg),
        comp_frac,
        yy/max(h-1,1),
        xx/max(w-1,1),
    ], axis=-1).astype(np.float32)
    names = [
        "prob", "blur_prob", "gray", "gray_minus_blur", "gray_gradient",
        "distance_to_base_skeleton", "distance_inside_base", "component_area_fraction",
        "normalized_y", "normalized_x",
    ]
    return X, names


def sample_keep_remove_training(image, probability, base_mask, gt_mask,
                                max_per_class: int = 10000,
                                seed: int = 1337) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from .action_targets import build_action_targets
    Xmap, names = build_removal_features(image, probability, base_mask)
    t = build_action_targets(base_mask, gt_mask)
    keep_idx = np.flatnonzero(t["keep"].ravel())
    rem_idx = np.flatnonzero(t["remove"].ravel())
    rng = np.random.default_rng(seed)
    if len(keep_idx) > max_per_class: keep_idx = rng.choice(keep_idx, max_per_class, replace=False)
    if len(rem_idx) > max_per_class: rem_idx = rng.choice(rem_idx, max_per_class, replace=False)
    idx = np.concatenate([keep_idx, rem_idx])
    y = np.concatenate([np.zeros(len(keep_idx), np.int8), np.ones(len(rem_idx), np.int8)])
    if len(idx) == 0: return np.empty((0, len(names)), np.float32), y, names
    return Xmap.reshape(-1, Xmap.shape[-1])[idx], y, names
