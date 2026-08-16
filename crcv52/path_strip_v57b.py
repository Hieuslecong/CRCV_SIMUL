from __future__ import annotations
import numpy as np, torch, cv2
from scipy.ndimage import distance_transform_edt
from .field_b import evidence
from .recovery import trace_inward
from .path_strip_v57 import (
    FAMILIES,unpack_add,ordered_path,resample_xy,_bilinear_map,geometry_plausibility,
    DW2d,SeqRes,same_source_listwise_loss,AcceptanceConfig,select_per_source
)
import torch.nn as nn
import torch.nn.functional as F

def build_path_strip_v57b(record:dict,row:dict,*,source_steps:int=8,path_steps:int=24,half_width:int=4):
    image=np.asarray(record['image'],np.float32); image=image/255. if image.max()>1.5 else image
    prob=np.asarray(record['prob'],np.float32);base=np.asarray(record['base'],bool);add=unpack_add(row);source=tuple(map(int,row['source_yx']))
    path=ordered_path(add,source)
    if len(path)<2:return None
    pxy=resample_xy(np.asarray([(x,y) for y,x in path],np.float32),path_steps)
    hist=trace_inward(base,source,8)
    if hist is None:
        v=pxy[min(2,len(pxy)-1)]-pxy[0];v=v/(np.linalg.norm(v)+1e-6);hxy=np.stack([pxy[0]-v*k for k in np.linspace(source_steps-1,0,source_steps)],0).astype(np.float32)
    else:hxy=resample_xy(np.asarray(hist,np.float32),source_steps)
    xy=np.vstack([hxy,pxy]);L=len(xy);t=np.empty_like(xy);t[1:-1]=xy[2:]-xy[:-2];t[0]=xy[1]-xy[0];t[-1]=xy[-1]-xy[-2];t/=np.linalg.norm(t,axis=1,keepdims=True)+1e-6;normal=np.c_[-t[:,1],t[:,0]]
    offs=np.arange(-half_width,half_width+1,dtype=np.float32);coords=xy[:,None,:]+normal[:,None,:]*offs[None,:,None]
    bh,dark,gr=evidence(image);ridge=np.maximum(bh,dark).astype(np.float32);dist=distance_transform_edt(~base).astype(np.float32);distn=np.clip(dist/8.,0,2)
    maps=np.dstack([image,np.clip(prob,0,1),base.astype(np.float32),np.clip(ridge,0,1),np.clip(gr,0,1),distn]).astype(np.float32)
    sampled=_bilinear_map(maps,coords)
    role=np.r_[np.zeros(source_steps,np.float32),np.ones(path_steps,np.float32)];pos=np.linspace(-1,1,L,dtype=np.float32)
    sampled=np.concatenate([sampled,role[:,None,None].repeat(sampled.shape[1],1),pos[:,None,None].repeat(sampled.shape[1],1)],axis=2)
    strip=torch.from_numpy(sampled.transpose(2,0,1)).float()
    fam=[1. if row['family']==f else 0. for f in FAMILIES];sim=geometry_plausibility(path)
    pp=np.rint(pxy).astype(int);pp[:,0]=np.clip(pp[:,0],0,base.shape[1]-1);pp[:,1]=np.clip(pp[:,1],0,base.shape[0]-1);dv=dist[pp[:,1],pp[:,0]]
    sv=hxy[-1]-hxy[-2];pv=pxy[1]-pxy[0];sv/=np.linalg.norm(sv)+1e-6;pv/=np.linalg.norm(pv)+1e-6;cos=float(np.clip(np.dot(sv,pv),-1,1))
    meta=torch.tensor([
        sim,np.log1p(float(row['length']))/np.log(25.),float(row['source_score']),float(row['score']),float(row['mean_ridge']),float(row['min_ridge']),float(row['mean_field']),float(row['min_field']),*fam,
        cos,float(np.clip(dv.max()/8.,0,2)),float(np.clip(dv[-1]/8.,0,2)),float(np.mean(dv>2.0)),float(np.clip((dv[-1]-dv[0])/8.,-2,2)),float(np.mean(dv<=1.5))
    ],dtype=torch.float32)
    return strip,meta,path

class PathAlignedStripVerifierV57B(nn.Module):
    def __init__(self,in_ch:int=10,meta_dim:int=17,c:int=44):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(in_ch,24,3,padding=1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),DW2d(24,c),DW2d(c,c));self.seq=nn.Sequential(SeqRes(c,1),SeqRes(c,2),SeqRes(c,4),SeqRes(c,8));self.att=nn.Conv1d(c,1,1);self.meta=nn.Sequential(nn.Linear(meta_dim,28),nn.LayerNorm(28),nn.SiLU());self.fuse=nn.Sequential(nn.Linear(c+28,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.10),nn.Linear(72,36),nn.SiLU());self.utility=nn.Linear(36,1);self.validity=nn.Linear(36,1);self.gain=nn.Linear(36,1)
    def forward(self,strip,meta):
        z=self.stem(strip).mean(-1);z=self.seq(z);a=torch.softmax(self.att(z).squeeze(1),dim=1);z=(z*a[:,None,:]).sum(-1);h=self.fuse(torch.cat([z,self.meta(meta)],1));return {'utility_logit':self.utility(h).squeeze(1),'validity_logit':self.validity(h).squeeze(1),'gain_logit':self.gain(h).squeeze(1)}

def v57b_loss(out,utility,validity,gain,groups):
    u=utility.float();v=validity.float();g=gain.float();posw=torch.tensor(min(max((len(u)-u.sum().item())/(u.sum().item()+1),1.),30.),device=u.device)
    b=F.binary_cross_entropy_with_logits(out['utility_logit'],u,pos_weight=posw);r=same_source_listwise_loss(out['utility_logit'],u,groups,hard_k=24);bv=F.binary_cross_entropy_with_logits(out['validity_logit'],v);rg=F.smooth_l1_loss(torch.sigmoid(out['gain_logit']),g)
    return b+1.8*r+.30*bv+.35*rg
