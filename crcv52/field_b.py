from __future__ import annotations
import cv2,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .field import DW,relprob,crop_spec,extract,place,PATCH,field_loss
class CenterlineFieldNetB(nn.Module):
    def __init__(self,c=16):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(12,c,3,1,1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),DW(c),DW(c,2),DW(c,3),nn.Conv2d(c,1,1))
    def forward(self,x):return self.net(x)[:,0]
def evidence(image):
    g=cv2.cvtColor((np.clip(image,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));blur=cv2.GaussianBlur(g,(0,0),2);dark=np.maximum(blur-g,0);gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3);gr=np.sqrt(gx*gx+gy*gy)
    def n(a):
        hi=float(np.quantile(a,.98));return np.clip(a/(hi+1e-6),0,1).astype(np.float32)
    return n(bh),n(dark),n(gr)
def candidate_tensor_b(image,prob,base,thr,q,gt=None,sigma=1.5,target_skeleton=None):
    y0,x0,ps=crop_spec(q['source_yx'],q['prior'],base.shape);corr=extract(q['corridor'].astype(np.uint8),y0,x0,ps).astype(bool);b=extract(base.astype(np.uint8),y0,x0,ps).astype(bool);im=extract(image.astype(np.float32),y0,x0,ps);rp=extract(relprob(prob,thr).astype(np.float32),y0,x0,ps)
    pm=np.zeros(base.shape,np.uint8);pts=np.vstack([np.array([[q['source_yx'][1],q['source_yx'][0]]],np.float32),q['prior']]);cv2.polylines(pm,[np.rint(pts).astype(np.int32)],False,1,1);pd=distance_transform_edt(~extract(pm,y0,x0,ps).astype(bool));pd=np.clip(pd/8.,0,1).astype(np.float32)
    src=np.zeros(base.shape,np.uint8);cv2.circle(src,(q['source_yx'][1],q['source_yx'][0]),2,1,-1);src=extract(src,y0,x0,ps).astype(np.float32);end=q['prior'][-1];dst=np.zeros(base.shape,np.uint8);cv2.circle(dst,(int(round(end[0])),int(round(end[1]))),2,1,-1);dst=extract(dst,y0,x0,ps).astype(np.float32);bh,dark,gr=evidence(image);bh=extract(bh,y0,x0,ps);dark=extract(dark,y0,x0,ps);gr=extract(gr,y0,x0,ps)
    X=np.dstack([im,rp,b.astype(np.float32),corr.astype(np.float32),pd,src,dst,bh,dark,gr]).transpose(2,0,1).astype(np.float32)
    if gt is None:return X,(y0,x0)
    gsk=skeletonize(gt.astype(bool)) if target_skeleton is None else target_skeleton.astype(bool);gskp=extract(gsk.astype(np.uint8),y0,x0,ps).astype(bool);dist=distance_transform_edt(~gskp);target=np.exp(-(dist**2)/(2*sigma*sigma)).astype(np.float32)*corr.astype(np.float32);target=np.maximum(target,.35*src*corr.astype(np.float32));return X,target,(y0,x0)
def infer_candidates_batch_b(model,image,prob,base,thr,qs,batch=128):
    if not qs:return []
    Xs=[];packs=[]
    for q in qs:
        X,(y0,x0)=candidate_tensor_b(image,prob,base,thr,q,None);Xs.append(X);packs.append((q,X,y0,x0))
    Fs=[]
    with torch.no_grad():
        for st in range(0,len(Xs),batch):Fs.extend(list(torch.sigmoid(model(torch.tensor(np.stack(Xs[st:st+batch]),dtype=torch.float32))).numpy()))
    from .recovery import route
    out=[]
    for (q,X,y0,x0),f in zip(packs,Fs):
        corr=X[5]>.5;b=X[4]>.5;pd=X[6];src=X[7]>.5;end=q['prior'][-1];ey=int(round(end[1]-y0));ex=int(round(end[0]-x0));term=np.zeros_like(corr,np.uint8)
        if 0<=ey<PATCH and 0<=ex<PATCH:cv2.circle(term,(ex,ey),6,1,-1)
        term=term.astype(bool)&corr&~b
        if not term.any():continue
        vals=np.where(term,f,-1);ty,tx=np.unravel_index(np.argmax(vals),vals.shape);target_score=float(f[ty,tx]);allowed=corr&~b;allowed|=src;sy,sx=q['source_yx'];ls=(sy-y0,sx-x0)
        if not(0<=ls[0]<PATCH and 0<=ls[1]<PATCH):continue
        allowed[ls]=1;cost=.80*(1-f)+.20*np.clip(pd,0,1);rr=route(cost.astype(np.float32),allowed,ls,(ty,tx))
        if rr is None or len(rr)<3:continue
        loc=np.zeros_like(corr,np.uint8)
        for y,x in rr:loc[y,x]=1
        full=place(loc,base.shape,y0,x0).astype(bool);add=full&~base
        if add.sum()<2:continue
        pv=f[loc.astype(bool)];ring=cv2.dilate(loc,np.ones((5,5),np.uint8),1).astype(bool)&corr&~loc.astype(bool);ring_mean=float(f[ring].mean()) if ring.any() else 0.;out.append({'add':add,'path':full,'score':.65*float(pv.mean())+.35*target_score,'mean_field':float(pv.mean()),'min_field':float(pv.min()),'target_score':target_score,'field_margin':float(pv.mean())-ring_mean,'candidate':q})
    return out

def infer_candidates_multitarget_b(model,image,prob,base,thr,qs,batch=128):
    if not qs:return []
    Xs=[];packs=[]
    for q in qs:
        X,(y0,x0)=candidate_tensor_b(image,prob,base,thr,q,None);Xs.append(X);packs.append((q,X,y0,x0))
    Fs=[]
    with torch.no_grad():
        for st in range(0,len(Xs),batch):Fs.extend(list(torch.sigmoid(model(torch.tensor(np.stack(Xs[st:st+batch]),dtype=torch.float32))).numpy()))
    from .recovery import route
    out=[]
    for (q,X,y0,x0),f in zip(packs,Fs):
        corr=X[5]>.5;b=X[4]>.5;pd=X[6];src=X[7]>.5;sy,sx=q['source_yx'];ls=(sy-y0,sx-x0)
        if not(0<=ls[0]<PATCH and 0<=ls[1]<PATCH):continue
        yy,xx=np.indices(corr.shape);ds=np.sqrt((yy-ls[0])**2+(xx-ls[1])**2);maxd=min(34.,float(q['horizon'])+8.);valid=corr&~b&(ds>=4)&(ds<=maxd);allowed=corr&~b;allowed|=src;allowed[ls]=1;cost=.80*(1-f)+.20*np.clip(pd,0,1)
        targets=[]
        for lo,hi in [(4,10),(10,18),(18,26),(26,35)]:
            reg=valid&(ds>=lo)&(ds<min(hi,maxd+1))
            if not reg.any():continue
            util=.82*f-.18*pd+.03*np.clip(ds/maxd,0,1);vals=np.where(reg,util,-1e9);ty,tx=np.unravel_index(np.argmax(vals),vals.shape);targets.append((int(ty),int(tx),float(f[ty,tx]),float(ds[ty,tx])))
        end=q['prior'][-1];ey=int(round(end[1]-y0));ex=int(round(end[0]-x0))
        if 0<=ey<PATCH and 0<=ex<PATCH and valid[ey,ex]:targets.append((ey,ex,float(f[ey,ex]),float(ds[ey,ex])))
        seen=set()
        for ty,tx,target_score,target_dist in targets:
            if (ty,tx) in seen:continue
            seen.add((ty,tx));allowed2=allowed.copy();allowed2[ty,tx]=1;rr=route(cost.astype(np.float32),allowed2,ls,(ty,tx))
            if rr is None or len(rr)<3:continue
            loc=np.zeros_like(corr,np.uint8)
            for y,x in rr:loc[y,x]=1
            full=place(loc,base.shape,y0,x0).astype(bool);add=full&~base
            if add.sum()<2:continue
            pv=f[loc.astype(bool)];ring=cv2.dilate(loc,np.ones((5,5),np.uint8),1).astype(bool)&corr&~loc.astype(bool);ring_mean=float(f[ring].mean()) if ring.any() else 0.;score=.55*float(pv.mean())+.25*target_score+.10*(float(pv.mean())-ring_mean)+.10*min(target_dist/24.,1.)
            qq=dict(q);qq['adaptive_target_yx']=(y0+ty,x0+tx);qq['adaptive_target_distance']=target_dist
            out.append({'add':add,'path':full,'score':score,'mean_field':float(pv.mean()),'min_field':float(pv.min()),'target_score':target_score,'field_margin':float(pv.mean())-ring_mean,'target_distance':target_dist,'candidate':qq})
    return out

def infer_candidates_multitarget_topk_b(model,image,prob,base,thr,qs,batch=128,targets_per_band=3,nms_radius=4):
    if not qs:return []
    Xs=[];packs=[]
    for q in qs:
        X,(y0,x0)=candidate_tensor_b(image,prob,base,thr,q,None);Xs.append(X);packs.append((q,X,y0,x0))
    Fs=[]
    with torch.no_grad():
        for st in range(0,len(Xs),batch):Fs.extend(list(torch.sigmoid(model(torch.tensor(np.stack(Xs[st:st+batch]),dtype=torch.float32))).numpy()))
    from .recovery import route
    out=[]
    for (q,X,y0,x0),f in zip(packs,Fs):
        corr=X[5]>.5;b=X[4]>.5;pd=X[6];src=X[7]>.5;sy,sx=q['source_yx'];ls=(sy-y0,sx-x0)
        if not(0<=ls[0]<PATCH and 0<=ls[1]<PATCH):continue
        yy,xx=np.indices(corr.shape);ds=np.sqrt((yy-ls[0])**2+(xx-ls[1])**2);maxd=min(34.,float(q['horizon'])+8.);valid=corr&~b&(ds>=4)&(ds<=maxd);allowed=corr&~b;allowed|=src;allowed[ls]=1;cost=.80*(1-f)+.20*np.clip(pd,0,1);util=.82*f-.18*pd+.03*np.clip(ds/max(maxd,1),0,1)
        targets=[]
        for lo,hi in [(4,10),(10,18),(18,26),(26,35)]:
            reg=valid&(ds>=lo)&(ds<min(hi,maxd+1))
            if not reg.any():continue
            work=np.where(reg,util,-1e9).astype(np.float32)
            for _ in range(int(targets_per_band)):
                ty,tx=np.unravel_index(np.argmax(work),work.shape);val=float(work[ty,tx])
                if val<-1e8:break
                targets.append((int(ty),int(tx),float(f[ty,tx]),float(ds[ty,tx])))
                cv2.circle(work,(int(tx),int(ty)),int(nms_radius),-1e9,-1)
        end=q['prior'][-1];ey=int(round(end[1]-y0));ex=int(round(end[0]-x0))
        if 0<=ey<PATCH and 0<=ex<PATCH and valid[ey,ex]:targets.append((ey,ex,float(f[ey,ex]),float(ds[ey,ex])))
        seen=set()
        for ty,tx,target_score,target_dist in targets:
            if (ty,tx) in seen:continue
            seen.add((ty,tx));allowed2=allowed.copy();allowed2[ty,tx]=1;rr=route(cost.astype(np.float32),allowed2,ls,(ty,tx))
            if rr is None or len(rr)<3:continue
            loc=np.zeros_like(corr,np.uint8)
            for y,x in rr:loc[y,x]=1
            full=place(loc,base.shape,y0,x0).astype(bool);add=full&~base
            if add.sum()<2:continue
            pv=f[loc.astype(bool)];ring=cv2.dilate(loc,np.ones((5,5),np.uint8),1).astype(bool)&corr&~loc.astype(bool);ring_mean=float(f[ring].mean()) if ring.any() else 0.;score=.55*float(pv.mean())+.25*target_score+.10*(float(pv.mean())-ring_mean)+.10*min(target_dist/24.,1.)
            qq=dict(q);qq['adaptive_target_yx']=(y0+ty,x0+tx);qq['adaptive_target_distance']=target_dist;qq['topk_target']=True
            out.append({'add':add,'path':full,'score':score,'mean_field':float(pv.mean()),'min_field':float(pv.min()),'target_score':target_score,'field_margin':float(pv.mean())-ring_mean,'target_distance':target_dist,'candidate':qq})
    return out
