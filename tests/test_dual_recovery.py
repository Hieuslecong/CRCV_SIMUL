import numpy as np, cv2
from crcv52.dual_recovery import EndpointInfo, compatible_pairs


def _info(ep,anchor,cid,score,v):
    h=np.array([[anchor[1]-v[0],anchor[0]-v[1]],[anchor[1],anchor[0]]],np.float32)
    return EndpointInfo(ep,anchor,h,cid,20,score,np.asarray(v,np.float32)/ (np.linalg.norm(v)+1e-8))


def test_pair_requires_different_components():
    a=_info((10,10),(10,10),1,.8,(1,0));b=_info((10,20),(10,20),1,.8,(-1,0))
    assert compatible_pairs([a,b])==[]


def test_face_to_face_pair_is_retained():
    a=_info((10,10),(10,10),1,.8,(1,0));b=_info((10,25),(10,25),2,.8,(-1,0))
    q=compatible_pairs([a,b])
    assert len(q)==1
    assert q[0]['facing_a']>.9 and q[0]['facing_b']>.9


def test_back_facing_pair_is_rejected():
    a=_info((10,10),(10,10),1,.8,(-1,0));b=_info((10,25),(10,25),2,.8,(1,0))
    assert compatible_pairs([a,b])==[]


def test_ridge_continue_symbol_exists():
    from crcv52.dual_recovery import ridge_continue_candidates
    assert callable(ridge_continue_candidates)


def test_bidirectional_meet_symbol_exists():
    from crcv52.dual_recovery import bidirectional_ridge_meet_candidates
    assert callable(bidirectional_ridge_meet_candidates)
