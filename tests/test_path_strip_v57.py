import numpy as np,torch
from crcv52.path_strip_v57 import resample_xy,geometry_plausibility,PathAlignedStripVerifier,same_source_listwise_loss,AcceptanceConfig,select_per_source

def test_resample_shape():
    p=np.array([[0,0],[1,0],[3,0]],np.float32);q=resample_xy(p,9);assert q.shape==(9,2);assert np.allclose(q[0],p[0]);assert np.allclose(q[-1],p[-1])
def test_geom_prefers_smooth():
    smooth=[(i,i) for i in range(10)];zig=[(i, i%2*5) for i in range(10)];assert geometry_plausibility(smooth)>geometry_plausibility(zig)
def test_net_shapes_and_budget():
    m=PathAlignedStripVerifier();o=m(torch.randn(3,9,32,9),torch.randn(3,11));assert o['utility_logit'].shape==(3,);assert sum(p.numel() for p in m.parameters())<150000
def test_rank_loss_positive():
    s=torch.tensor([.2,.8,.4]);y=torch.tensor([0.,1.,0.]);g=torch.tensor([0,0,0]);assert float(same_source_listwise_loss(s,y,g))>0
def test_acceptance_top1_margin():
    rows=[{'image':'a','source_yx':(1,1),'source_score':.8,'length':8},{'image':'a','source_yx':(1,1),'source_score':.8,'length':8}]
    cfg=AcceptanceConfig(.8,.8,.1);assert select_per_source(rows,[.95,.7],[.95,.95],cfg)==[0]
