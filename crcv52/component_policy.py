from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ComponentRemovalConfig:
    """Conservative whole-component fallback for atomic detached-like predictions."""
    max_area_fraction: float = 0.001
    max_mean_probability: float = 0.85
    max_pixels: int = 16


def component_remove_mask(base_mask, probability,
                          config: ComponentRemovalConfig | None = None) -> np.ndarray:
    """Return a GT-free whole-component REMOVE mask.

    Only complete small Base components are eligible; partial edge shaving is
    impossible by construction. Parameters must be selected on CAL only.
    """
    cfg = config or ComponentRemovalConfig()
    base = np.asarray(base_mask, bool)
    prob = np.asarray(probability, np.float32)
    if base.shape != prob.shape or base.ndim != 2:
        raise ValueError("bad base/probability shapes")
    labels, n = ndi.label(base)
    removed = np.zeros_like(base)
    for lab in range(1, n + 1):
        reg = labels == lab
        size = int(reg.sum())
        if size == 0 or size > cfg.max_pixels:
            continue
        if size / float(base.size) > cfg.max_area_fraction:
            continue
        if float(prob[reg].mean()) > cfg.max_mean_probability:
            continue
        removed |= reg
    return removed
