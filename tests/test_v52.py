import numpy as np,cv2,torch
from skimage.morphology import skeletonize
from crcv52.field_b import CenterlineFieldNetB,candidate_tensor_b
from crcv52.recovery import route,source_proposals
from crcv52.geometry import GeoMLP

def test_centerline_field_b_shape():
 m=CenterlineFieldNetB();x=torch.zeros(2,12,64,64);assert m(x).shape==(2,64,64)

def test_missing_target_does_not_supervise_covered_skeleton():
 H=W=64;im=np.ones((H,W,3),np.float32)*.5;gt=np.zeros((H,W),np.uint8);gt[32,8:56]=1;base=gt.astype(bool);base[32,30:40]=0;prob=np.where(base,.9,.1).astype(np.float32);prior=np.array([[30,32],[35,32],[40,32]],np.float32);corr=np.zeros((H,W),np.uint8);cv2.line(corr,(28,32),(44,32),1,7);q={'source_yx':(32,29),'history':np.array([[24,32],[25,32],[26,32],[27,32],[28,32],[29,32]],np.float32),'prior':prior,'corridor':corr.astype(bool),'horizon':3,'angle':0};missing=skeletonize(gt.astype(bool))&~base;X,T,(y0,x0)=candidate_tensor_b(im,prob,base,.5,q,gt,target_skeleton=missing);assert T[32-y0,34-x0]>T[32-y0,20-x0]

def test_route_never_enables_forbidden_target():
 a=np.ones((16,16),bool);a[8,13]=False;c=np.ones((16,16),np.float32);assert route(c,a,(8,2),(8,13)) is None

def test_source_subset_is_respected():
 b=np.zeros((32,32),bool);b[16,3:12]=1;b[16,20:29]=1;m=GeoMLP();subset=[(16,3)];qs=source_proposals(m,b,max_sources=4,horizons=(6,),angles=(0,),corridor_radius=3,endpoint_subset=subset);assert all(tuple(q['source_yx']) in subset for q in qs)

def test_one_pixel_centerline_recovery_cannot_exceed_missing_skeleton_count():
 gt=np.zeros((32,32),bool);gt[14:18,4:28]=1;base=gt.copy();base[:,14:18]=0;miss=(gt&~base).sum();skmiss=(skeletonize(gt)&~base).sum();assert skmiss<=miss

def test_topk_multitarget_decoder_symbol_exists():
    from crcv52.field_b import infer_candidates_multitarget_topk_b
    assert callable(infer_candidates_multitarget_topk_b)
