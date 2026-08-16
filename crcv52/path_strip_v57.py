from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize
from .field_b import evidence
from .recovery import trace_inward

FAMILIES=("v52b_topk","iterative","ridge_continue")

def unpack_add(row:dict)->np.ndarray:
    bits=np.frombuffer(row['add_pack'],np.uint8); n=int(np.prod(row['shape']))
    return np.unpackbits(bits)[:n].reshape(row['shape']).astype(bool)

def ordered_path(add:np.ndarray, source_yx:tuple[int,int])->list[tuple[int,int]]:
    """Longest connected skeleton route starting nearest the source endpoint."""
    m=np.asarray(add,bool); sk=m if int(m.sum())<=128 else skeletonize(m); pts=np.argwhere(sk)
    if not len(pts): return []
    sy,sx=map(int,source_yx); d2=(pts[:,0]-sy)**2+(pts[:,1]-sx)**2
    start=tuple(map(int,pts[int(np.argmin(d2))])); S={tuple(map(int,p)) for p in pts}
    q=deque([start]);par={start:None};dist={start:0};far=start
    while q:
        y,x=q.popleft()
        if dist[(y,x)]>dist[far]:far=(y,x)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not(dy or dx):continue
                nb=(y+dy,x+dx)
                if nb in S and nb not in par:
                    par[nb]=(y,x);dist[nb]=dist[(y,x)]+1;q.append(nb)
    out=[];cur=far
    while cur is not None:out.append(cur);cur=par[cur]
    return out[::-1]

def resample_xy(xy:np.ndarray,n:int)->np.ndarray:
    p=np.asarray(xy,np.float32)
    if len(p)==0:return np.zeros((n,2),np.float32)
    if len(p)==1:return np.repeat(p,n,axis=0)
    d=np.linalg.norm(np.diff(p,axis=0),axis=1); keep=np.r_[True,d>1e-6];p=p[keep]
    if len(p)==1:return np.repeat(p,n,axis=0)
    d=np.linalg.norm(np.diff(p,axis=0),axis=1);c=np.r_[0.,np.cumsum(d)];tot=float(c[-1])
    if tot<1e-6:return np.repeat(p[:1],n,axis=0)
    q=np.linspace(0,tot,n,dtype=np.float32)
    return np.c_[np.interp(q,c,p[:,0]),np.interp(q,c,p[:,1])].astype(np.float32)

def _bilinear_map(stack:np.ndarray,xy:np.ndarray)->np.ndarray:
    H,W=stack.shape[:2]; shp=xy.shape[:-1]
    mx=np.clip(xy[...,0],0,W-1).astype(np.float32);my=np.clip(xy[...,1],0,H-1).astype(np.float32)
    z=cv2.remap(stack,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    if z.ndim==2:z=z[...,None]
    return z.reshape(*shp,stack.shape[2]).astype(np.float32)

def geometry_plausibility(path_yx:list[tuple[int,int]])->float:
    """Weak geometry score; only a prior, never an acceptance decision."""
    if len(path_yx)<4:return .5
    p=resample_xy(np.asarray([(x,y) for y,x in path_yx],np.float32),min(32,max(8,len(path_yx))))
    d=np.diff(p,axis=0);n=np.linalg.norm(d,axis=1);u=d/(n[:,None]+1e-6)
    ang=np.arccos(np.clip((u[:-1]*u[1:]).sum(1),-1,1)) if len(u)>1 else np.zeros(1)
    turn=float(np.median(ang)); tort=float(n.sum()/(np.linalg.norm(p[-1]-p[0])+1e-6))
    st=math.exp(-0.5*(turn/0.40)**2); sr=math.exp(-0.5*((max(tort,1)-1)/0.45)**2)
    return float(np.clip(.75*st+.25*sr,0,1))

def build_path_strip(record:dict,row:dict,*,source_steps:int=8,path_steps:int=24,half_width:int=4):
    image=np.asarray(record['image'],np.float32); image=image/255. if image.max()>1.5 else image
    prob=np.asarray(record['prob'],np.float32);base=np.asarray(record['base'],bool);add=unpack_add(row)
    source=tuple(map(int,row['source_yx'])); path=ordered_path(add,source)
    if len(path)<2: return None
    pxy=resample_xy(np.asarray([(x,y) for y,x in path],np.float32),path_steps)
    hist=trace_inward(base,source,8)
    if hist is None:
        v=pxy[min(2,len(pxy)-1)]-pxy[0];v=v/(np.linalg.norm(v)+1e-6)
        hxy=np.stack([pxy[0]-v*k for k in np.linspace(source_steps-1,0,source_steps)],0).astype(np.float32)
    else:hxy=resample_xy(np.asarray(hist,np.float32),source_steps)
    xy=np.vstack([hxy,pxy]); L=len(xy)
    t=np.empty_like(xy);t[1:-1]=xy[2:]-xy[:-2];t[0]=xy[1]-xy[0];t[-1]=xy[-1]-xy[-2]
    t/=np.linalg.norm(t,axis=1,keepdims=True)+1e-6; normal=np.c_[-t[:,1],t[:,0]]
    offsets=np.arange(-half_width,half_width+1,dtype=np.float32)
    coords=xy[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    bh,dark,gr=evidence(image);ridge=np.maximum(bh,dark).astype(np.float32)
    maps=np.dstack([image,np.clip(prob,0,1),base.astype(np.float32),np.clip(ridge,0,1),np.clip(gr,0,1)]).astype(np.float32)
    sampled=_bilinear_map(maps,coords)
    role=np.r_[np.zeros(source_steps,np.float32),np.ones(path_steps,np.float32)]
    pos=np.linspace(-1,1,L,dtype=np.float32)
    sampled=np.concatenate([sampled,role[:,None,None].repeat(sampled.shape[1],1),pos[:,None,None].repeat(sampled.shape[1],1)],axis=2)
    strip=torch.from_numpy(sampled.transpose(2,0,1)).float()
    fam=[1. if row['family']==f else 0. for f in FAMILIES]
    sim=geometry_plausibility(path)
    meta=torch.tensor([sim,np.log1p(float(row['length']))/np.log(25.),float(row['source_score']),float(row['score']),float(row['mean_ridge']),float(row['min_ridge']),float(row['mean_field']),float(row['min_field']),*fam],dtype=torch.float32)
    return strip,meta,path

class DW2d(nn.Module):
    def __init__(self,ci,co):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(ci,ci,3,1,1,groups=ci,bias=False),nn.Conv2d(ci,co,1,bias=False),nn.GroupNorm(4,co),nn.SiLU())
    def forward(self,x):return self.net(x)
class SeqRes(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.dw=nn.Conv1d(c,c,3,padding=d,dilation=d,groups=c,bias=False);self.pw=nn.Conv1d(c,c,1,bias=False);self.gn=nn.GroupNorm(4,c)
    def forward(self,x):return x+F.silu(self.gn(self.pw(self.dw(x))))
class PathAlignedStripVerifier(nn.Module):
    """V5.7 ordered-path verifier. Preserves along-path evidence instead of square resizing."""
    def __init__(self,in_ch:int=9,meta_dim:int=11,c:int=40):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(in_ch,24,3,padding=1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),DW2d(24,c),DW2d(c,c))
        self.seq=nn.Sequential(SeqRes(c,1),SeqRes(c,2),SeqRes(c,4),SeqRes(c,8))
        self.att=nn.Conv1d(c,1,1)
        self.meta=nn.Sequential(nn.Linear(meta_dim,24),nn.LayerNorm(24),nn.SiLU())
        self.fuse=nn.Sequential(nn.Linear(c+24,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,32),nn.SiLU())
        self.utility=nn.Linear(32,1);self.validity=nn.Linear(32,1);self.gain=nn.Linear(32,1)
    def forward(self,strip,meta):
        z=self.stem(strip).mean(-1);z=self.seq(z);a=torch.softmax(self.att(z).squeeze(1),dim=1);z=(z*a[:,None,:]).sum(-1)
        h=self.fuse(torch.cat([z,self.meta(meta)],1))
        return {'utility_logit':self.utility(h).squeeze(1),'validity_logit':self.validity(h).squeeze(1),'gain_logit':self.gain(h).squeeze(1)}

def same_source_listwise_loss(scores,labels,groups,*,hard_k:int=16):
    terms=[]
    for g in torch.unique(groups):
        m=groups==g; s=scores[m]; y=labels[m]>0.5
        if not y.any() or (~y).sum()==0:continue
        pos=s[y];neg=s[~y]; neg=torch.topk(neg,min(hard_k,len(neg))).values
        terms.append(F.softplus(.35-pos[:,None]+neg[None,:]).mean())
    return torch.stack(terms).mean() if terms else scores.sum()*0.

def v57_loss(out,utility,validity,gain,groups):
    u=utility.float();v=validity.float();g=gain.float()
    posw=torch.tensor(min(max((len(u)-u.sum().item())/(u.sum().item()+1),1.),30.),device=u.device)
    b=F.binary_cross_entropy_with_logits(out['utility_logit'],u,pos_weight=posw)
    r=same_source_listwise_loss(out['utility_logit'],u,groups)
    bv=F.binary_cross_entropy_with_logits(out['validity_logit'],v)
    rg=F.smooth_l1_loss(torch.sigmoid(out['gain_logit']),g)
    return b+1.2*r+.35*bv+.25*rg

@dataclass(frozen=True)
class AcceptanceConfig:
    utility_threshold:float
    validity_threshold:float
    margin_threshold:float
    source_threshold:float=.35
    max_length:int=24
    max_accept_per_image:int=2

def select_per_source(rows,scores,validity,cfg:AcceptanceConfig):
    """GT-free Top-1 + relative-margin + abstention selection."""
    groups={}
    for i,r in enumerate(rows):
        if float(r['source_score'])<cfg.source_threshold or float(r['length'])>cfg.max_length:continue
        groups.setdefault((r['image'],tuple(r['source_yx'])),[]).append(i)
    accepted=[]
    for _,ids in groups.items():
        ids=sorted(ids,key=lambda i:float(scores[i]),reverse=True);i=ids[0];s1=float(scores[i]);s2=float(scores[ids[1]]) if len(ids)>1 else 0.
        if s1>=cfg.utility_threshold and float(validity[i])>=cfg.validity_threshold and s1-s2>=cfg.margin_threshold:accepted.append(i)
    byimg={}
    for i in accepted:byimg.setdefault(rows[i]['image'],[]).append(i)
    out=[]
    for im,ids in byimg.items():out+=sorted(ids,key=lambda i:float(scores[i]),reverse=True)[:cfg.max_accept_per_image]
    return out
