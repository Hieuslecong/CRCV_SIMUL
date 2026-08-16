from __future__ import annotations
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from .field import relprob

def descriptor(image,prob,base,thr,z):
    add=z['add'].astype(bool);path=z['path'].astype(bool);q=z['candidate'];g=cv2.cvtColor((np.clip(image,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3);gr=np.sqrt(gx*gx+gy*gy);rp=relprob(prob,thr);sy,sx=q['source_yx'];n,lab=cv2.connectedComponents(base.astype(np.uint8),8);cid=int(lab[sy,sx]);comp=lab==cid
    def st(a,r):
        v=a[r]
        if not len(v):return [0.,0.,0.,0.]
        return [float(v.mean()),float(v.std()),float(np.quantile(v,.25)),float(np.quantile(v,.75))]
    src=np.zeros_like(base,np.uint8);cv2.circle(src,(sx,sy),4,1,-1);src=src.astype(bool)&base;corr=q['corridor']&~base
    f=[z['score'],z['mean_field'],z['min_field'],z['target_score'],float(z.get('field_margin',0.)),min(add.sum()/32.,3.),q['horizon']/24.,abs(q['angle'])/8.,np.log1p(comp.sum())/8.]
    for a in [rp,bh,gr,g]:f+=st(a,add)+st(a,src)
    # relational differences source <-> trajectory and path-vs-corridor contrast
    for a in [rp,bh,gr,g]:f.append(abs(st(a,add)[0]-st(a,src)[0]));f.append(st(a,add)[0]-st(a,corr)[0])
    # geometry/path agreement
    ys,xs=np.where(path)
    if len(ys)>=2:
        v=np.array([xs[-1]-xs[0],ys[-1]-ys[0]],float);hv=q['history'][-1]-q['history'][-2];compat=float(np.dot(v,hv)/(np.linalg.norm(v)*np.linalg.norm(hv)+1e-9));straight=float(np.linalg.norm(v)/max(len(ys),1))
    else:compat=straight=0.
    f += [compat,straight]
    return np.asarray(f,np.float32)

def label_candidate(gt,base,z):
    gt=gt.astype(bool);add=z['add'].astype(bool);tp=int((add&gt).sum());fp=int((add&~gt).sum());prec=tp/(tp+fp+1e-9)
    from skimage.morphology import skeletonize
    skm=skeletonize(gt)&~base;tol=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);sh=int((tol&skm).sum())
    if prec>=.85 and sh>=2:return 1,{'precision':prec,'tp':tp,'fp':fp,'skeleton_hit':sh}
    if prec<=.20 or sh==0:return 0,{'precision':prec,'tp':tp,'fp':fp,'skeleton_hit':sh}
    return -1,{'precision':prec,'tp':tp,'fp':fp,'skeleton_hit':sh}
