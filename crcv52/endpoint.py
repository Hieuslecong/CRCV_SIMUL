from __future__ import annotations
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .recovery import endpoints,trace_inward,degree
from .field import relprob

def rgb_maps(image):
 g=cv2.cvtColor((np.clip(image,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3);gr=np.sqrt(gx*gx+gy*gy);return g,bh,gr

def features(image,prob,base,thr,ep,hist=None):
    if hist is None:hist=trace_inward(base,ep,6)
    if hist is None:return None
    g,bh,gr=rgb_maps(image);rp=relprob(prob,thr);y,x=ep;n,lab=cv2.connectedComponents(base.astype(np.uint8),8);cid=int(lab[y,x]);comp=lab==cid;sk=skeletonize(comp);dist=distance_transform_edt(base)
    r=5;y0=max(0,y-r);y1=min(base.shape[0],y+r+1);x0=max(0,x-r);x1=min(base.shape[1],x+r+1);reg=np.zeros_like(base,bool);reg[y0:y1,x0:x1]=1
    vals=lambda a:[float(a[reg].mean()),float(a[reg].std())]
    v=hist[-1]-hist[-2];# outward vector is last history direction
    vv=v/(np.linalg.norm(v)+1e-8);ahead=[]
    for d in range(2,11):
        xx=int(round(x+vv[0]*d));yy=int(round(y+vv[1]*d))
        if 0<=yy<base.shape[0] and 0<=xx<base.shape[1]:ahead.append((yy,xx))
    av=lambda a:float(np.mean([a[q] for q in ahead])) if ahead else 0.
    return np.asarray([np.log1p(comp.sum())/8.,np.log1p(sk.sum())/8.,float(dist[y,x]/5.),*vals(rp),*vals(bh),*vals(gr),*vals(g),av(rp),av(bh),av(gr),av(g)],np.float32)

def label(gt,base,ep,hist=None):
    if hist is None:hist=trace_inward(base,ep,6)
    if hist is None:return -1
    sk=skeletonize(gt.astype(bool));missing=sk&~base;y,x=ep;v=hist[-1]-hist[-2];vv=v/(np.linalg.norm(v)+1e-8);hit=0
    for d in range(2,16):
        xx=int(round(x+vv[0]*d));yy=int(round(y+vv[1]*d))
        if 0<=yy<base.shape[0] and 0<=xx<base.shape[1]:
            y0=max(0,yy-1);y1=min(base.shape[0],yy+2);x0=max(0,xx-1);x1=min(base.shape[1],xx+2);hit+=int(missing[y0:y1,x0:x1].any())
    return 1 if hit>=2 else 0
