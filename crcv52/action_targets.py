from __future__ import annotations

import numpy as np


def build_action_targets(base_mask, gt_mask) -> dict[str, np.ndarray]:
    """Build exact corrective-action targets from a frozen Base mask and GT.

    KEEP   = Base ∩ GT
    REMOVE = Base \ GT
    ADD    = GT \ Base
    IGNORE = outside Base and outside GT
    """
    base = np.asarray(base_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if base.shape != gt.shape:
        raise ValueError("base_mask and gt_mask must have identical shapes")
    keep = base & gt
    remove = base & ~gt
    add = gt & ~base
    ignore = ~base & ~gt
    if np.any(keep & remove) or np.any(keep & add) or np.any(remove & add):
        raise AssertionError("action targets overlap")
    return {"keep": keep, "remove": remove, "add": add, "ignore": ignore}


def encode_actions(base_mask, gt_mask) -> np.ndarray:
    """Return int8 action map: 0=IGNORE, 1=KEEP, 2=REMOVE, 3=ADD."""
    t = build_action_targets(base_mask, gt_mask)
    out = np.zeros(np.asarray(base_mask).shape, dtype=np.int8)
    out[t["keep"]] = 1
    out[t["remove"]] = 2
    out[t["add"]] = 3
    return out
