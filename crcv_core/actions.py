from __future__ import annotations
import numpy as np

IGNORE=np.int8(0); KEEP=np.int8(1); REMOVE=np.int8(2); ADD=np.int8(3)

def _mask(x,name):
    a=np.asarray(x,bool)
    if a.ndim!=2: raise ValueError(f"{name} must be 2-D")
    return a

def action_targets(base_mask,gt_mask):
    """Exact training-only actions from frozen Base and GT."""
    b=_mask(base_mask,"base_mask"); g=_mask(gt_mask,"gt_mask")
    if b.shape!=g.shape: raise ValueError("base_mask and gt_mask must match")
    keep=b&g; remove=b&~g; add=g&~b; ignore=~b&~g
    if not np.array_equal(keep|remove,b): raise AssertionError("Base partition failure")
    if not np.array_equal(keep|add,g): raise AssertionError("GT partition failure")
    return {"keep":keep,"remove":remove,"add":add,"ignore":ignore}
