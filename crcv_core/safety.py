from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

@dataclass(frozen=True)
class SafetyConfig:
    core_radius_fraction: float=.85
    min_radius_norm: float=.002
    max_total_remove_fraction: float=.04
    max_foreground_remove_fraction: float=.25
    max_region_fraction: float=.03
    max_add_foreground_fraction: float=.20
    preserve_component_count: bool=True

def _check(c):
    for n in ("core_radius_fraction","min_radius_norm","max_total_remove_fraction","max_foreground_remove_fraction","max_region_fraction","max_add_foreground_fraction"):
        x=float(getattr(c,n))
        if not np.isfinite(x) or x<0 or (n!="core_radius_fraction" and x>1):raise ValueError(f"bad {n}")

def project_remove(base_mask,score,threshold,config:SafetyConfig|None=None):
    c=config or SafetyConfig(); _check(c); b=np.asarray(base_mask,bool); s=np.asarray(score,np.float32)
    if b.ndim!=2 or s.shape!=b.shape:raise ValueError("bad base/score shapes")
    if not np.isfinite(s).all() or not np.isfinite(threshold):raise ValueError("non-finite score/threshold")
    if not b.any():return np.zeros_like(b),{"removed_pixels":0,"status":"NO_OP_EMPTY"}
    diag=max(float(np.hypot(*b.shape)),1.); sk=skeletonize(b); inside=ndi.distance_transform_edt(b).astype(np.float32)
    if sk.any():
        dist,idx=ndi.distance_transform_edt(~sk,return_indices=True); local=np.maximum(inside[idx[0],idx[1]],1.); lim=np.maximum(c.min_radius_norm*diag,c.core_radius_fraction*local); protected=b&(dist<=lim)
    else:protected=np.zeros_like(b)
    if not 0<=float(threshold)<=1:raise ValueError("threshold must be in [0,1]")
    conn=np.ones((3,3),bool); cand=b&~protected&(s>=float(threshold)); lab,n=ndi.label(cand,structure=conn); regions=[]
    for i in range(1,n+1):
        r=lab==i; regions.append((-float(s[r].mean()),int(r.sum()),i,r))
    regions.sort(key=lambda z:(z[0],z[1],z[2])); budget=max(0,min(int(c.max_total_remove_fraction*b.size),int(c.max_foreground_remove_fraction*int(b.sum())))); cap=max(0,int(c.max_region_fraction*b.size)); rm=np.zeros_like(b); used=0; nb=int(ndi.label(b,structure=conn)[1])
    for _,size,_,r in regions:
        if size>cap or used+size>budget:continue
        trial=b&~(rm|r)
        if c.preserve_component_count and int(ndi.label(trial,structure=conn)[1])>nb:continue
        rm|=r; used+=size
    if np.any(rm&~b) or np.any(rm&protected):raise AssertionError("REMOVE safety invariant")
    if sk.any() and np.any(rm&sk):raise AssertionError("Base skeleton removed")
    return rm,{"removed_pixels":int(rm.sum()),"budget":int(budget),"status":"PASS"}

def project_add(base_mask,candidate,score,threshold,config:SafetyConfig|None=None):
    c=config or SafetyConfig(); _check(c); b=np.asarray(base_mask,bool); cand=np.asarray(candidate,bool); s=np.asarray(score,np.float32)
    if b.ndim!=2 or cand.shape!=b.shape or s.shape!=b.shape:raise ValueError("bad ADD shapes")
    if not np.isfinite(s).all() or not np.isfinite(threshold):raise ValueError("non-finite ADD score/threshold")
    if not 0<=float(threshold)<=1:raise ValueError("threshold must be in [0,1]")
    conn=np.ones((3,3),bool); raw=cand&(s>=float(threshold))
    if not raw.any() or not b.any():return np.zeros_like(b),{"added_pixels":0,"status":"NO_OP"}
    lab,_=ndi.label(b|raw,structure=conn); good=np.unique(lab[b]); add=raw&np.isin(lab,good); budget=max(0,int(c.max_add_foreground_fraction*int(b.sum())))
    if budget==0:return np.zeros_like(b),{"added_pixels":0,"budget":0,"status":"NO_OP_BUDGET"}
    if int(add.sum())>budget:
        idx=np.flatnonzero(add.ravel()); vals=s.ravel()[idx]; keep=idx[np.argpartition(vals,-budget)[-budget:]]; q=np.zeros_like(add); q.ravel()[keep]=True
        lab,_=ndi.label(b|q,structure=conn); good=np.unique(lab[b]); add=q&np.isin(lab,good)
    if np.any(add&b) or np.any(add&~cand):raise AssertionError("ADD safety invariant")
    return add,{"added_pixels":int(add.sum()),"budget":int(budget),"status":"PASS"}
