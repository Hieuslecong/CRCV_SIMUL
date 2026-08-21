from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ComponentRemovalConfig:
    """Scale-free conservative whole-component REMOVE fallback.

    All size limits are fractions of the current mask/foreground. No absolute
    pixel-count threshold is used, so the same configuration has comparable
    semantics at 128, 256, or another evaluation resolution.
    """
    max_area_fraction: float = 0.001
    max_mean_probability: float = 0.85
    max_total_remove_fraction: float = 0.03
    max_foreground_remove_fraction: float = 0.10


def component_remove_mask(base_mask, probability,
                          config: ComponentRemovalConfig | None = None) -> np.ndarray:
    """Return a GT-free, scale-free whole-component REMOVE mask.

    Only complete Base components are removed, therefore partial edge shaving is
    impossible by construction. Candidate components are considered from lower to
    higher mean Base probability and accepted only while both image-area and
    foreground-area budgets remain satisfied. Parameters must be selected without
    TEST data.
    """
    cfg = config or ComponentRemovalConfig()
    base = np.asarray(base_mask, bool)
    prob = np.asarray(probability, np.float32)
    if base.shape != prob.shape or base.ndim != 2:
        raise ValueError("bad base/probability shapes")
    if not np.isfinite(prob).all():
        raise ValueError("probability contains non-finite values")
    if not base.any():
        return np.zeros_like(base)

    labels, n = ndi.label(base)
    candidates = []
    for lab in range(1, n + 1):
        reg = labels == lab
        size = int(reg.sum())
        if size == 0:
            continue
        area_fraction = size / float(base.size)
        mean_probability = float(prob[reg].mean())
        if area_fraction > cfg.max_area_fraction:
            continue
        if mean_probability > cfg.max_mean_probability:
            continue
        candidates.append((mean_probability, area_fraction, lab, reg))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    image_budget = max(1, int(np.floor(cfg.max_total_remove_fraction * base.size)))
    foreground_budget = max(1, int(np.floor(cfg.max_foreground_remove_fraction * int(base.sum()))))
    budget = min(image_budget, foreground_budget)

    removed = np.zeros_like(base)
    used = 0
    for _, _, _, reg in candidates:
        size = int(reg.sum())
        if used + size > budget:
            continue
        removed |= reg
        used += size
    return removed
