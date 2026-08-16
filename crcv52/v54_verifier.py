from __future__ import annotations
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from .field import relprob
from .field_b import evidence

FAMS=('v52b','iterative','ridge_continue','bidir_connect')

def prepare_context(record):
    image=record['image'].astype(np.float32);prob=record['prob'].astype(np.float32);base=record['base'].astype(bool);thr=float(record['threshold'])
    bh,dark,gr=evidence(image);gray=cv2.cvtColor((np.clip(image,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;rp=relprob(prob,thr);n,lab=cv2.connectedComponents(base.astype(np.uint8),8)
    return {'image':image,'prob':prob,'base':base,'thr':thr,'maps':(rp,bh,dark,gr,gray),'labels':lab,'foreign_dist':{}}

def _stats(a,m):
    v=a[m]
    if not len(v):return (0.,0.)
    return float(v.mean()),float(v.std())

def source_key(fam,z):
    if fam=='v52b':return ('s',tuple(map(int,z['candidate']['source_yx'])))
    if fam=='iterative':return ('s',tuple(map(int,z.get('source_endpoint_yx',z.get('source_yx',(0,0))))))
    if fam=='ridge_continue':return ('s',tuple(map(int,z.get('source_endpoint_yx',z.get('source_yx',(0,0))))))
    a=tuple(map(int,z.get('source_endpoint_yx',z.get('source_yx',(0,0)))));b=tuple(map(int,z.get('destination_endpoint_yx',z.get('destination_yx',(0,0)))));return ('p',)+tuple(sorted((a,b)))

def descriptor(record,fam,z,ctx=None):
    c=prepare_context(record) if ctx is None else ctx;base=c['base'];lab=c['labels'];add=z['add'].astype(bool);path=add;ys,xs=np.where(path)
    # source/endpoints
    if fam=='v52b':
        sy,sx=map(int,z['candidate']['source_yx']);hist=np.asarray(z['candidate']['history'],np.float32)
    else:
        sy,sx=map(int,z.get('source_yx',(0,0)));hist=np.asarray(z.get('hist_xy',[]),np.float32)
    src=np.zeros_like(base,np.uint8)
    if 0<=sy<base.shape[0] and 0<=sx<base.shape[1]:cv2.circle(src,(sx,sy),4,1,-1)
    src=src.astype(bool)&base
    ring=cv2.dilate(add.astype(np.uint8),np.ones((5,5),np.uint8),1).astype(bool)&~add
    # common scalar fields, missing values are 0 by schema.
    raw_score=float(z.get('score',z.get('pair_score',0.)))
    mean_field=float(z.get('mean_field',z.get('mean_pair_field',0.)))
    min_field=float(z.get('min_field',z.get('min_pair_field',0.)))
    mean_ridge=float(z.get('mean_ridge',0.));min_ridge=float(z.get('min_ridge',0.))
    length=float(z.get('length',int(add.sum())))
    source_score=float(z.get('source_score',0.));dest_score=float(z.get('destination_score',0.))
    pair_score=float(z.get('pair_score',0.));pair_dist=float(z.get('pair_distance',0.));fa=float(z.get('facing_a',0.));fb=float(z.get('facing_b',0.))
    identity=float(z.get('min_identity',0.));stop=float(z.get('stop_prob',0.));field_margin=float(z.get('field_margin',0.))
    fam_one=[1. if fam==f else 0. for f in FAMS]
    # path geometry
    pxy=np.asarray([(x,y) for y,x in z.get('path_yx',[])],np.float32)
    if len(pxy)<2 and len(xs)>=2:pxy=np.c_[xs,ys].astype(np.float32)
    straight=mean_turn=max_turn=0.
    if len(pxy)>=2:
        st=np.diff(pxy,axis=0);norm=st/(np.linalg.norm(st,axis=1,keepdims=True)+1e-8);straight=float(np.linalg.norm(pxy[-1]-pxy[0])/max(len(pxy)-1,1))
        if len(norm)>=2:
            ang=np.arccos(np.clip(np.sum(norm[:-1]*norm[1:],axis=1),-1,1));mean_turn=float(ang.mean()/np.pi);max_turn=float(ang.max()/np.pi)
    # topology relation: distance from path endpoint to foreign Base; contact count.
    cid=int(lab[sy,sx]) if 0<=sy<base.shape[0] and 0<=sx<base.shape[1] else 0
    if cid not in c['foreign_dist']:
        own=lab==cid if cid>0 else np.zeros_like(base);foreign=base&~own;c['foreign_dist'][cid]=distance_transform_edt(~foreign).astype(np.float32) if foreign.any() else np.full(base.shape,99.,np.float32)
    fd=c['foreign_dist'][cid];end_yx=(int(ys[-1]),int(xs[-1])) if len(ys) else (sy,sx);end_dist=float(fd[end_yx]);touch=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool)&base;ids={int(v) for v in np.unique(lab[touch]) if int(v)>0 and int(v)!=cid}
    f=fam_one+[raw_score,mean_field,min_field,field_margin,mean_ridge,min_ridge,identity,stop,source_score,dest_score,pair_score,min(pair_dist/64.,2.),fa,fb,min(length/64.,2.),straight,mean_turn,max_turn,min(end_dist/32.,3.),min(len(ids)/3.,2.),float(bool(ids)),float(fam=='bidir_connect')]
    for a in c['maps']:
        pm,ps=_stats(a,add);sm,ss=_stats(a,src);rm,rs=_stats(a,ring);f += [pm,ps,sm,ss,pm-sm,pm-rm]
    return np.asarray(f,np.float32)

def add_relative_features(records):
    """Append same-source relative features without labels."""
    groups={}
    for i,r in enumerate(records):groups.setdefault(r['source_key'],[]).append(i)
    keys=('raw_score','evidence','length','source_score')
    for ids in groups.values():
        vals={k:np.asarray([records[i][k] for i in ids],np.float32) for k in keys}
        for i in ids:
            rel=[]
            for k in keys:
                v=float(records[i][k]);a=vals[k];med=float(np.median(a));sd=float(a.std()+1e-6);rank=float((a<=v).mean());rel += [(v-med)/sd,rank]
            records[i]['feature']=np.r_[records[i]['feature'],np.asarray(rel,np.float32)]
    return records
