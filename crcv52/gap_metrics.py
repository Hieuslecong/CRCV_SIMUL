from __future__ import annotations
import cv2,numpy as np
from skimage.morphology import skeletonize

def core_missing_skeleton(gt,base,clearance=2):
    """GT skeleton pixels representing a true topological gap, not 1-2px alignment error."""
    gsk=skeletonize(gt.astype(bool));b=base.astype(bool)
    if clearance<=0:return gsk&~b
    k=2*int(clearance)+1;near=cv2.dilate(b.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1).astype(bool)
    return gsk&~near

def core_gap_hit(add,gt,base,clearance=2,tolerance=1):
    core=core_missing_skeleton(gt,base,clearance)
    a=add.astype(bool)
    if tolerance>0:
        k=2*int(tolerance)+1;a=cv2.dilate(a.astype(np.uint8),np.ones((k,k),np.uint8),1).astype(bool)
    hit=a&core
    return int(hit.sum()),core,hit
