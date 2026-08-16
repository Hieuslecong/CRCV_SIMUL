import torch
from crcv52.path_strip_v57b import PathAlignedStripVerifierV57B

def test_v57b_shapes_budget():
 m=PathAlignedStripVerifierV57B();o=m(torch.randn(2,10,32,9),torch.randn(2,17));assert o['utility_logit'].shape==(2,);assert sum(p.numel() for p in m.parameters())<30000
