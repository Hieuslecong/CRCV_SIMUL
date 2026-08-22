from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

@dataclass(frozen=True)
class FeatureConfig:
    blur_sigma_norm: float=.006
    blackhat_size_norm: float=.02

def _robust01(a):
    a=np.asarray(a,np.float32)
    if not a.size:return a
    lo=float(np.quantile(a,.02)); hi=float(np.quantile(a,.98))
    if not np.isfinite(lo+hi) or hi<=lo+1e-8:return np.zeros_like(a,np.float32)
    return np.clip((a-lo)/(hi-lo+1e-6),0,1).astype(np.float32)

def build_features(image,probability,base_mask,config:FeatureConfig|None=None):
    """Nine compact GT-free features shared by ADD and REMOVE heads."""
    c=config or FeatureConfig(); im=np.asarray(image,np.float32); p=np.asarray(probability,np.float32); b=np.asarray(base_mask,bool)
    if b.ndim!=2 or p.shape!=b.shape or im.ndim!=3 or im.shape[:2]!=b.shape or im.shape[2]<1 or min(b.shape)<2:raise ValueError("bad image/probability/base shapes")
    if not np.isfinite(im).all() or not np.isfinite(p).all():raise ValueError("non-finite inputs")
    if im.size and (float(im.min())<0 or float(im.max())>1):raise ValueError("image must be normalized to [0,1]")
    if p.size and (float(p.min())<0 or float(p.max())>1):raise ValueError("probability must be in [0,1]")
    if c.blur_sigma_norm<0 or c.blackhat_size_norm<0 or not np.isfinite(c.blur_sigma_norm+c.blackhat_size_norm):raise ValueError("bad feature config")
    h,w=b.shape; diag=max(float(np.hypot(h,w)),1.); sigma=max(.5,float(c.blur_sigma_norm)*diag)
    gray=im.mean(2); blur=ndi.gaussian_filter(gray,sigma); pblur=ndi.gaussian_filter(p,sigma); gy,gx=np.gradient(gray); grad=np.sqrt(gx*gx+gy*gy)
    k=max(3,int(round(float(c.blackhat_size_norm)*diag))|1); blackhat=_robust01(ndi.grey_closing(gray,size=(k,k))-gray)
    sk=skeletonize(b); inside=ndi.distance_transform_edt(b).astype(np.float32)
    if sk.any():
        dist,idx=ndi.distance_transform_edt(~sk,return_indices=True); local=np.maximum(inside[idx[0],idx[1]],1.); radial=np.clip(dist/local,0,4).astype(np.float32)
        neigh=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode="constant"); degree=neigh[idx[0],idx[1]].astype(np.float32)/9.
    else: radial=np.zeros(b.shape,np.float32); degree=np.zeros(b.shape,np.float32)
    X=np.stack([p,pblur,gray,gray-blur,_robust01(grad),blackhat,radial,np.clip(inside/diag,0,1),degree],-1).astype(np.float32)
    names=["prob","blur_prob","gray","local_contrast","gradient","blackhat","radial_ratio","inside_radius_norm","nearest_skeleton_degree"]
    return X,names
