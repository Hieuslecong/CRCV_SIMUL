from __future__ import annotations
import torch,torch.nn as nn,torch.nn.functional as F
from .field import DW
class CenterlineFieldUNet(nn.Module):
    """V5.2d global-context field head for 64x64 candidate patches."""
    def __init__(self):
        super().__init__();self.e1=nn.Sequential(nn.Conv2d(12,16,3,1,1,bias=False),nn.GroupNorm(4,16),nn.SiLU(),DW(16));self.down2=nn.Sequential(nn.Conv2d(16,24,3,2,1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),DW(24));self.down3=nn.Sequential(nn.Conv2d(24,32,3,2,1,bias=False),nn.GroupNorm(4,32),nn.SiLU(),DW(32),DW(32,2));self.d2=nn.Sequential(nn.Conv2d(32+24,24,1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),DW(24));self.d1=nn.Sequential(nn.Conv2d(24+16,16,1,bias=False),nn.GroupNorm(4,16),nn.SiLU(),DW(16));self.o=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.down2(a);c=self.down3(b);z=F.interpolate(c,b.shape[-2:],mode='bilinear',align_corners=False);z=self.d2(torch.cat([z,b],1));z=F.interpolate(z,a.shape[-2:],mode='bilinear',align_corners=False);z=self.d1(torch.cat([z,a],1));return self.o(z)[:,0]
