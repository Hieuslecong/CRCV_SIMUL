import numpy as np
from crcv52.suppression_v55c import (
    enumerate_runtime_components,
    confidence_suppression_mask,
    calibrate_confidence_suppression,
)


def _record(include_gt=True):
    base=np.zeros((16,16),np.uint8);base[2:4,2:5]=1;base[10:12,10:13]=1
    prob=np.zeros((16,16),np.float32);prob[2:4,2:5]=.9;prob[10:12,10:13]=.55
    r={'base':base,'prob':prob}
    if include_gt:
        gt=np.zeros_like(base);gt[2:4,2:5]=1;r['gt']=gt
    return r


def test_runtime_enumeration_does_not_require_gt():
    z=enumerate_runtime_components(_record(False))
    assert len(z)==2
    assert sorted(round(x.mean_probability,2) for x in z)==[.55,.9]


def test_runtime_suppression_operates_without_gt():
    r=_record(False);rm=confidence_suppression_mask(r,mean_probability_threshold=.6)
    assert int(rm.sum())==6
    assert not rm[2:4,2:5].any()
    assert rm[10:12,10:13].all()


def test_calibration_and_runtime_share_all_components():
    best,ok=calibrate_confidence_suppression([_record(True)],max_true_pixel_removal=.01,target_false_pixel_removal=.9)
    assert ok
    assert best['true_pixel_removal']==0.0
    assert best['false_pixel_removal']>0.99
    rm=confidence_suppression_mask(_record(False),mean_probability_threshold=best['mean_probability_threshold'])
    assert int(rm.sum())==6
