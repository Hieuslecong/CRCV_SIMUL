from __future__ import annotations
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from .field import relprob
from .field_b import evidence
from .recovery import trace_inward

def _ms(a,m):
    v=a[m]
    if not len(v):return (0.,0.)
    return float(v.mean()),float(v.std())

def prepare_context(record):
    image=record['image'].astype(np.float32);prob=record['prob'].astype(np.float32);base=record['base'].astype(bool);thr=float(record['threshold']);rp=relprob(prob,thr);bh,dark,gr=evidence(image);gray=cv2.cvtColor((np.clip(image,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;_,lab=cv2.connectedComponents(base.astype(np.uint8),8);return {'base':base,'maps':[rp,bh,dark,gr,gray],'labels':lab,'fd_cache':{},'hist_cache':{},'src_cache':{}}

def descriptor(record,z,ctx=None):
    c=prepare_context(record) if ctx is None else ctx;base=c['base'];H,W=base.shape;path=z['add'].astype(bool);ys,xs=np.where(path);sy,sx=map(int,z['source_yx']);sep=tuple(map(int,z.get('source_endpoint_yx',(sy,sx))))
    if sep not in c['hist_cache']:c['hist_cache'][sep]=trace_inward(base,sep,6)
    hist=c['hist_cache'][sep];src_key=(sy,sx)
    if src_key not in c['src_cache']:
        src=np.zeros_like(base,np.uint8);cv2.circle(src,(sx,sy),4,1,-1);c['src_cache'][src_key]=src.astype(bool)&base
    src=c['src_cache'][src_key];ring=cv2.dilate(path.astype(np.uint8),np.ones((5,5),np.uint8),1).astype(bool)&~path;lab=c['labels'];cid=int(lab[sy,sx]) if 0<=sy<H and 0<=sx<W else 0;comp=lab==cid if cid>0 else np.zeros_like(base)
    if cid not in c['fd_cache']:
        foreign=base&~comp;c['fd_cache'][cid]=distance_transform_edt(~foreign).astype(np.float32) if foreign.any() else None
    fd=c['fd_cache'][cid];end=(int(ys[-1]),int(xs[-1])) if len(ys) else (sy,sx);start_dist=float(fd[sy,sx]) if fd is not None else 99.;end_dist=float(fd[end]) if fd is not None else 99.;approach=np.clip((start_dist-end_dist)/20.,-3,3);pxy=np.asarray([(x,y) for y,x in z.get('path_yx',[])],np.float32);straight=mean_turn=max_turn=start_compat=end_compat=0.
    if len(pxy)>=2:
        steps=np.diff(pxy,axis=0);norm=steps/(np.linalg.norm(steps,axis=1,keepdims=True)+1e-8);straight=float(np.linalg.norm(pxy[-1]-pxy[0])/max(len(pxy)-1,1))
        if len(norm)>=2:
            turns=np.arccos(np.clip(np.sum(norm[:-1]*norm[1:],axis=1),-1,1));mean_turn=float(turns.mean()/np.pi);max_turn=float(turns.max()/np.pi)
        if hist is not None and len(hist)>=2:
            sv=hist[-1]-hist[-2];sv/=np.linalg.norm(sv)+1e-8;start_compat=float(np.dot(sv,norm[0]));end_compat=float(np.dot(sv,norm[-1]))
    f=[float(z['score']),float(z['min_identity']),float(z['stop_prob']),float(z['mean_field']),float(z['min_field']),float(z['source_score']),min(float(z['length'])/32.,2.),float(z.get('connects_foreign',False)),float(z.get('learned_stop',False)),approach,min(end_dist/30.,3.),min(float(comp.sum())/5000.,3.),straight,mean_turn,max_turn,start_compat,end_compat,float(z.get('consensus_mean',0.)),float(z.get('consensus_min',0.)),min(float(z.get('source_candidate_count',0))/64.,3.),float(z.get('rel_score',0.)),float(z.get('rel_min_identity',0.)),float(z.get('rel_stop_prob',0.)),float(z.get('rel_mean_field',0.)),float(z.get('rel_min_field',0.)),float(z.get('rel_length',0.))]
    for a in c['maps']:
        pm,ps=_ms(a,path);sm,ss=_ms(a,src);rm,rs=_ms(a,ring);f += [pm,ps,sm,ss,pm-sm,pm-rm]
    return np.asarray(f,np.float32)
