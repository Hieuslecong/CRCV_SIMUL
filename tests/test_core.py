import numpy as np,cv2
from crcv52.recovery import route,corridor_from_prior,trace_inward

def test_route_rejects_forbidden_target():
 a=np.ones((12,12),bool);a[6,9]=False;c=np.ones((12,12),np.float32);assert route(c,a,(6,2),(6,9)) is None

def test_corridor_contains_prior():
 p=np.array([[2,5],[9,5]],np.float32);c=corridor_from_prior(p,(12,12),2);assert all(c[int(y),int(x)] for x,y in p)

def test_trace_inward_line():
 m=np.zeros((32,32),np.uint8);cv2.line(m,(4,16),(25,16),1,1);h=trace_inward(m,(16,4),6);assert h is not None and len(h)==6
