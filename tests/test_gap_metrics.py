import numpy as np,cv2
from crcv52.gap_metrics import core_missing_skeleton

def test_alignment_offset_is_not_core_gap():
    gt=np.zeros((32,32),np.uint8);base=np.zeros_like(gt)
    cv2.line(gt,(5,16),(26,16),1,1);cv2.line(base,(5,17),(26,17),1,1)
    assert core_missing_skeleton(gt,base,clearance=2).sum()==0

def test_real_gap_interior_survives_clearance():
    gt=np.zeros((32,32),np.uint8);base=np.zeros_like(gt)
    cv2.line(gt,(4,16),(27,16),1,1);base[16,4:10]=1;base[16,22:28]=1
    core=core_missing_skeleton(gt,base,clearance=2)
    assert core.sum()>5
    assert core[16,15]
