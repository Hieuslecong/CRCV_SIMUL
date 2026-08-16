from __future__ import annotations
from dataclasses import dataclass
import cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

PATCH=64
class DW(nn.Module):
    def __init__(self,c,d=1):super().__init__();self.dw=nn.Conv2d(c,c,3,1,d,dilation=d,groups=c,bias=False);self.pw=nn.Conv2d(c,c,1,bias=False);self.g=nn.GroupNorm(4,c)
    def forward(self,x):return F.silu(self.g(self.pw(self.dw(x))))
class CenterlineFieldNet(nn.Module):
    def __init__(self,c=16):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(9,c,3,1,1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),DW(c),DW(c,2),DW(c,3),nn.Conv2d(c,1,1))
    def forward(self,x):return self.net(x)[:,0]

def relprob(prob,thr):
    p=np.clip(prob,1e-5,1-1e-5);t=np.clip(thr,1e-5,1-1e-5);z=np.log(p/(1-p))-np.log(t/(1-t));return 1/(1+np.exp(-np.clip(z,-12,12)))
def crop_spec(source_yx,prior,shape,patch=PATCH):
    sy,sx=source_yx;pts=np.vstack([np.array([[sx,sy]],np.float32),np.asarray(prior,np.float32)]);cx=int(round(float(pts[:,0].mean())));cy=int(round(float(pts[:,1].mean())));h=patch//2;return cy-h,cx-h,patch
def extract(a,y0,x0,patch=PATCH,fill=0):
    if a.ndim==2:out=np.full((patch,patch),fill,a.dtype)
    else:out=np.full((patch,patch,a.shape[2]),fill,a.dtype)
    H,W=a.shape[:2];ya=max(0,y0);yb=min(H,y0+patch);xa=max(0,x0);xb=min(W,x0+patch);oy=ya-y0;ox=xa-x0
    if ya<yb and xa<xb:out[oy:oy+(yb-ya),ox:ox+(xb-xa)]=a[ya:yb,xa:xb]
    return out
def place(local,shape,y0,x0):
    H,W=shape;out=np.zeros(shape,local.dtype);ya=max(0,y0);yb=min(H,y0+local.shape[0]);xa=max(0,x0);xb=min(W,x0+local.shape[1]);oy=ya-y0;ox=xa-x0
    if ya<yb and xa<xb:out[ya:yb,xa:xb]=local[oy:oy+(yb-ya),ox:ox+(xb-xa)]
    return out
def candidate_tensor(image,prob,base,thr,q,gt=None,sigma=1.5,target_skeleton=None):
    y0,x0,ps=crop_spec(q['source_yx'],q['prior'],base.shape);corr=extract(q['corridor'].astype(np.uint8),y0,x0,ps).astype(bool);b=extract(base.astype(np.uint8),y0,x0,ps).astype(bool);im=extract(image.astype(np.float32),y0,x0,ps);rp=extract(relprob(prob,thr).astype(np.float32),y0,x0,ps)
    pm=np.zeros(base.shape,np.uint8);pts=np.vstack([np.array([[q['source_yx'][1],q['source_yx'][0]]],np.float32),q['prior']]);cv2.polylines(pm,[np.rint(pts).astype(np.int32)],False,1,1);pd=distance_transform_edt(~extract(pm,y0,x0,ps).astype(bool));pd=np.clip(pd/8.,0,1).astype(np.float32)
    src=np.zeros(base.shape,np.uint8);cv2.circle(src,(q['source_yx'][1],q['source_yx'][0]),2,1,-1);src=extract(src,y0,x0,ps).astype(np.float32)
    end=q['prior'][-1];dst=np.zeros(base.shape,np.uint8);cv2.circle(dst,(int(round(end[0])),int(round(end[1]))),2,1,-1);dst=extract(dst,y0,x0,ps).astype(np.float32)
    X=np.dstack([im,rp,b.astype(np.float32),corr.astype(np.float32),pd,src,dst]).transpose(2,0,1).astype(np.float32)
    if gt is None:return X,(y0,x0)
    gsk=skeletonize(gt.astype(bool)) if target_skeleton is None else target_skeleton.astype(bool)
    gskp=extract(gsk.astype(np.uint8),y0,x0,ps).astype(bool);dist=distance_transform_edt(~gskp);target=np.exp(-(dist**2)/(2*sigma*sigma)).astype(np.float32)*corr.astype(np.float32)
    target=np.maximum(target,.35*src* corr.astype(np.float32))
    return X,target,(y0,x0)
def field_loss(logit,target):
    w=1+6*target;b=F.binary_cross_entropy_with_logits(logit,target,weight=w)
    p=torch.sigmoid(logit);hi=target>.7;lo=target<.2
    if hi.any() and lo.any():
        hp=p[hi];lp=p[lo];n=min(hp.numel(),lp.numel(),2048);rank=F.relu(.25-hp[:n]+lp[:n]).mean()
    else:rank=logit.sum()*0
    return b+.5*rank

def infer_candidate(model,image,prob,base,thr,q):
    X,(y0,x0)=candidate_tensor(image,prob,base,thr,q,None)
    with torch.no_grad():f=torch.sigmoid(model(torch.tensor(X[None],dtype=torch.float32)))[0].numpy()
    corr=X[5]>.5;b=X[4]>.5;pd=X[6];src=X[7]>.5
    end=q['prior'][-1];ey=int(round(end[1]-y0));ex=int(round(end[0]-x0));term=np.zeros_like(corr,np.uint8)
    if 0<=ey<PATCH and 0<=ex<PATCH:cv2.circle(term,(ex,ey),6,1,-1)
    term=term.astype(bool)&corr&~b
    if not term.any():return None
    vals=np.where(term,f,-1);ty,tx=np.unravel_index(np.argmax(vals),vals.shape);target_score=float(f[ty,tx])
    allowed=corr&~b;allowed|=src
    from .recovery import route
    sy,sx=q['source_yx'];ls=(sy-y0,sx-x0)
    if not (0<=ls[0]<PATCH and 0<=ls[1]<PATCH):return None
    allowed[ls]=True
    cost=.80*(1-f)+.20*np.clip(pd,0,1)
    rr=route(cost.astype(np.float32),allowed,ls,(ty,tx))
    if rr is None or len(rr)<3:return None
    local=np.zeros_like(corr,np.uint8)
    for y,x in rr:local[y,x]=1
    full=place(local,base.shape,y0,x0).astype(bool);add=full&~base
    if int(add.sum())<2:return None
    pathvals=f[local.astype(bool)];score=.65*float(pathvals.mean())+.35*target_score
    return {'add':add,'path':full,'score':score,'mean_field':float(pathvals.mean()),'min_field':float(pathvals.min()),'target_score':target_score,'field_local':f,'crop_origin':(y0,x0),'candidate':q}

def infer_candidates_batch(model,image,prob,base,thr,qs,batch=128):
    if not qs:return []
    packs=[];Xs=[]
    for q in qs:
        X,(y0,x0)=candidate_tensor(image,prob,base,thr,q,None);Xs.append(X);packs.append((q,X,y0,x0))
    Fs=[]
    with torch.no_grad():
        for st in range(0,len(Xs),batch):
            z=torch.sigmoid(model(torch.tensor(np.stack(Xs[st:st+batch]),dtype=torch.float32))).numpy();Fs.extend(list(z))
    from .recovery import route
    out=[]
    for (q,X,y0,x0),f in zip(packs,Fs):
        corr=X[5]>.5;b=X[4]>.5;pd=X[6];src=X[7]>.5;end=q['prior'][-1];ey=int(round(end[1]-y0));ex=int(round(end[0]-x0));term=np.zeros_like(corr,np.uint8)
        if 0<=ey<PATCH and 0<=ex<PATCH:cv2.circle(term,(ex,ey),6,1,-1)
        term=term.astype(bool)&corr&~b
        if not term.any():continue
        vals=np.where(term,f,-1);ty,tx=np.unravel_index(np.argmax(vals),vals.shape);target_score=float(f[ty,tx]);allowed=corr&~b;allowed|=src;sy,sx=q['source_yx'];ls=(sy-y0,sx-x0)
        if not (0<=ls[0]<PATCH and 0<=ls[1]<PATCH):continue
        allowed[ls]=True;cost=.80*(1-f)+.20*np.clip(pd,0,1);rr=route(cost.astype(np.float32),allowed,ls,(ty,tx))
        if rr is None or len(rr)<3:continue
        local=np.zeros_like(corr,np.uint8)
        for y,x in rr:local[y,x]=1
        full=place(local,base.shape,y0,x0).astype(bool);add=full&~base
        if int(add.sum())<2:continue
        pv=f[local.astype(bool)];score=.65*float(pv.mean())+.35*target_score;out.append({'add':add,'path':full,'score':score,'mean_field':float(pv.mean()),'min_field':float(pv.min()),'target_score':target_score,'candidate':q})
    return out
