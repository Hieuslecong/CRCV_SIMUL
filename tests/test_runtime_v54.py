import numpy as np
from crcv52.runtime_v54 import CRCVV54Block,V54RuntimeConfig

def test_v54_default_runtime_is_fail_closed():
    b=np.zeros((8,8),np.uint8);b[2:4,3]=1
    out,d=CRCVV54Block().refine(b)
    assert np.array_equal(out,b)
    assert d['proposal_qualified'] is True
    assert d['verifier_qualified'] is False
    assert d['recovery_applied'] is False

def test_v54_rejects_unqualified_recovery_enable():
    c=V54RuntimeConfig(True,True,False,False,False,False,'FAIL_CLOSED_BASE_ONLY')
    try:CRCVV54Block(c);assert False
    except RuntimeError:pass
