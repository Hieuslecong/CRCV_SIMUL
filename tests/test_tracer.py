import numpy as np, torch
from crcv52.tracer import IterativeTracerNet,dir_class,STOP_CLASS,state_tensor

def test_tracer_shapes():
 m=IterativeTracerNet();x=torch.zeros(3,12,21,21);a,b=m(x);assert a.shape==(3,9) and b.shape==(3,)

def test_direction_classes_complete():
 cur=(10,10);seen=set()
 for dy in (-1,0,1):
  for dx in (-1,0,1):
   if dy==0 and dx==0:continue
   seen.add(dir_class(cur,(10+dy,10+dx)))
 assert seen==set(range(8)) and STOP_CLASS==8

def test_boundary_exit_anchor_reaches_component_boundary():
    import cv2, numpy as np
    from crcv52.recovery import boundary_exit_anchor
    m=np.zeros((41,41),np.uint8);cv2.line(m,(6,20),(25,20),1,7)
    ep=(20,22);hist=np.asarray([[17,20],[18,20],[19,20],[20,20],[21,20],[22,20]],np.float32)
    a,h=boundary_exit_anchor(m.astype(bool),ep,hist,12)
    assert m[a]>0
    y,x=a
    assert any(0<=y+dy<41 and 0<=x+dx<41 and not m[y+dy,x+dx] for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)])
    assert tuple(np.rint(h[-1]).astype(int))==(a[1],a[0])


def test_rank_valid_actions_after_legality_filter():
    import cv2
    from crcv52.tracer import ranked_valid_moves,DIRS,STOP_CLASS
    base=np.zeros((9,9),bool);base[3:6,3:6]=1
    _,labels=cv2.connectedComponents(base.astype(np.uint8),8)
    cur=(4,5);corr=np.ones_like(base,bool);path=[cur]
    # Give all illegal in-component moves higher log-probability than the only legal exit.
    lp=np.full(9,-20.,np.float32);lp[STOP_CLASS]=-1
    for i,(dy,dx) in enumerate(DIRS):
        y,x=cur[0]+dy,cur[1]+dx
        lp[i]=0.-i*.01 if base[y,x] else -5.-i*.01
    moves=ranked_valid_moves(lp,cur,base,labels,int(labels[cur]),corr,path,3)
    assert moves
    assert all(not base[y,x] for _,y,x in moves)

def test_v53_runtime_is_fail_closed():
    from crcv52.runtime_v53 import CRCVV53Block
    b=np.zeros((12,12),np.uint8);b[3:8,5]=1
    out,diag=CRCVV53Block().refine(b)
    assert np.array_equal(out,b)
    assert diag["recovery_applied"] is False


def test_source_field_atlas_api_exists():
    from crcv52.tracer import source_field_atlas
    assert callable(source_field_atlas)
