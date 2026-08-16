from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .endpoint import features as endpoint_features
from .field import relprob
from .field_b import evidence
from .recovery import endpoints, trace_inward, boundary_exit_anchor, route
from .tracer import source_field_map


@dataclass
class EndpointInfo:
    endpoint_yx: tuple[int,int]
    anchor_yx: tuple[int,int]
    history_xy: np.ndarray
    component_id: int
    component_area: int
    source_score: float
    outward_xy: np.ndarray


def _outward(hist: np.ndarray) -> np.ndarray:
    h=np.asarray(hist,np.float32)
    if len(h)<2:return np.array([1.,0.],np.float32)
    v=h[-1]-h[-2]
    n=float(np.linalg.norm(v))
    return (v/(n+1e-8)).astype(np.float32)


def ranked_endpoint_infos(record, ranker, *, border_margin=10, per_component=2, max_sources=16):
    """Component-balanced endpoint ranking.

    At most `per_component` endpoints are retained from each component before a global
    top-K. This prevents a noisy multi-endpoint blob from monopolizing the source pool.
    """
    im=record['image'].astype(np.float32);p=record['prob'].astype(np.float32);b=record['base'].astype(bool);thr=float(record['threshold'])
    H,W=b.shape;n,lab=cv2.connectedComponents(b.astype(np.uint8),8);areas=np.bincount(lab.ravel(),minlength=n)
    by={}
    for ep in endpoints(b):
        y,x=map(int,ep)
        if y<border_margin or x<border_margin or y>=H-border_margin or x>=W-border_margin:continue
        hist=trace_inward(b,ep,6)
        if hist is None:continue
        f=endpoint_features(im,p,b,thr,ep,hist)
        if f is None:continue
        sc=float(ranker.predict_proba(f[None])[:,1][0]) if ranker is not None else 0.
        anchor,h2=boundary_exit_anchor(b,ep,hist,12)
        cid=int(lab[y,x])
        info=EndpointInfo((y,x),tuple(map(int,anchor)),np.asarray(h2,np.float32),cid,int(areas[cid]),sc,_outward(h2))
        by.setdefault(cid,[]).append(info)
    pool=[]
    for cid,v in by.items():
        v=sorted(v,key=lambda z:z.source_score,reverse=True)[:per_component];pool.extend(v)
    return sorted(pool,key=lambda z:z.source_score,reverse=True)[:max_sources]


def compatible_pairs(infos, *, min_distance=4., max_distance=64., facing_min=-0.35, facing_sum_min=-0.15, max_pairs=48):
    """Enumerate destination-conditioned CONNECT hypotheses without GT.

    Pair endpoints must belong to different Base components. The outward tangents need not
    point perfectly at each other (curved gaps are allowed), but grossly back-facing pairs
    are removed. The score is only a proposal-prior; no pair is claimed correct here.
    """
    rows=[]
    for i,a in enumerate(infos):
        ay,ax=a.anchor_yx
        for b in infos[i+1:]:
            if a.component_id==b.component_id:continue
            by,bx=b.anchor_yx
            v=np.array([bx-ax,by-ay],np.float32);d=float(np.linalg.norm(v))
            if d<min_distance or d>max_distance:continue
            uv=v/(d+1e-8)
            fa=float(np.dot(a.outward_xy,uv));fb=float(np.dot(b.outward_xy,-uv))
            if fa<facing_min or fb<facing_min or fa+fb<facing_sum_min:continue
            dist_prior=math.exp(-((d-20.)/28.)**2)
            score=.34*(a.source_score+b.source_score)+.22*(fa+fb)+.10*dist_prior
            rows.append({'a':a,'b':b,'distance':d,'facing_a':fa,'facing_b':fb,'pair_score':float(score)})
    return sorted(rows,key=lambda z:z['pair_score'],reverse=True)[:max_pairs]


def _line_corridor(shape,a_yx,b_yx,radius):
    m=np.zeros(shape,np.uint8);cv2.line(m,(int(a_yx[1]),int(a_yx[0])),(int(b_yx[1]),int(b_yx[0])),1,1)
    k=2*int(radius)+1
    return cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1).astype(bool)


def _prior_distance(shape,qa,qb,a_yx,b_yx,scale):
    m=np.zeros(shape,np.uint8)
    for q,anchor in [(qa,a_yx),(qb,b_yx)]:
        pts=np.vstack([np.array([[anchor[1],anchor[0]]],np.float32),np.asarray(q['prior'],np.float32)])
        cv2.polylines(m,[np.rint(pts).astype(np.int32)],False,1,1,lineType=cv2.LINE_8)
    cv2.line(m,(int(a_yx[1]),int(a_yx[0])),(int(b_yx[1]),int(b_yx[0])),1,1)
    d=distance_transform_edt(~m.astype(bool)).astype(np.float32)
    return np.clip(d/max(float(scale),1.),0,1)


def _path_mask(path,shape):
    m=np.zeros(shape,np.uint8)
    for y,x in path:m[int(y),int(x)]=1
    return m.astype(bool)


def bidirectional_connect_candidates(field_model,geo,image,prob,base,thr,pair, *, corridor_radius=7):
    """Destination-conditioned CONNECT proposals using evidence from both endpoints.

    This is deliberately proposal-only. It uses no GT. Both source-conditioned centerline
    fields are combined with RGB ridge evidence and a bidirectional geometry prior. Multiple
    fixed cost mixtures are emitted so the downstream verifier/oracle can assess coverage.
    """
    base=base.astype(bool);a:EndpointInfo=pair['a'];b:EndpointInfo=pair['b'];ay,ax=a.anchor_yx;by,bx=b.anchor_yx
    horizon=int(np.clip(math.ceil(pair['distance'])+6,12,36))
    fa,qa=source_field_map(field_model,geo,image,prob,base,thr,a.anchor_yx,a.history_xy,horizon,corridor_radius)
    fb,qb=source_field_map(field_model,geo,image,prob,base,thr,b.anchor_yx,b.history_xy,horizon,corridor_radius)
    tube_r=int(np.clip(7+pair['distance']*.12,8,15));tube=_line_corridor(base.shape,a.anchor_yx,b.anchor_yx,tube_r)
    corr=tube & (qa['corridor']|qb['corridor']|_line_corridor(base.shape,a.anchor_yx,b.anchor_yx,max(5,tube_r//2)))
    corr |= _line_corridor(base.shape,a.anchor_yx,b.anchor_yx,max(5,tube_r//2))
    allowed=corr & ~base
    allowed[ay,ax]=True;allowed[by,bx]=True
    bh,dark,gr=evidence(image);ridge=np.maximum(bh,dark);rp=relprob(prob,thr)
    pair_field=np.sqrt(np.clip(fa,0,1)*np.clip(fb,0,1)).astype(np.float32)
    avg_field=.5*(np.clip(fa,0,1)+np.clip(fb,0,1))
    gd=_prior_distance(base.shape,qa,qb,a.anchor_yx,b.anchor_yx,tube_r)
    variants={
      'balanced': .35*(1-pair_field)+.20*(1-avg_field)+.20*(1-ridge)+.15*gd+.10*(1-rp),
      'field_heavy': .55*(1-pair_field)+.15*(1-ridge)+.20*gd+.10*(1-rp),
      'rgb_heavy': .25*(1-pair_field)+.15*(1-avg_field)+.35*(1-ridge)+.15*gd+.10*(1-rp),
    }
    out=[];seen=set()
    for name,cost in variants.items():
        cost=np.asarray(cost,np.float32);cost[~corr]=1e6
        rr=route(cost,allowed,(ay,ax),(by,bx))
        if rr is None or len(rr)<3:continue
        path=_path_mask(rr,base.shape);add=path&~base
        if int(add.sum())<2:continue
        key=np.packbits(add).tobytes()
        if key in seen:continue
        seen.add(key)
        vals=pair_field[add] if add.any() else np.array([0.],np.float32);rvals=ridge[add] if add.any() else np.array([0.],np.float32)
        out.append({'family':'v54_bidir_connect','variant':name,'add':add,'path':path,'path_yx':rr,'source_yx':a.anchor_yx,'destination_yx':b.anchor_yx,
                    'source_endpoint_yx':a.endpoint_yx,'destination_endpoint_yx':b.endpoint_yx,'source_component':a.component_id,'destination_component':b.component_id,
                    'source_score':a.source_score,'destination_score':b.source_score,'pair_score':pair['pair_score'],'pair_distance':pair['distance'],
                    'facing_a':pair['facing_a'],'facing_b':pair['facing_b'],'mean_pair_field':float(vals.mean()),'min_pair_field':float(vals.min()),
                    'mean_ridge':float(rvals.mean()),'corridor':corr,'learned_stop':True,'connects_foreign':True,'length':int(add.sum())})
    return out


def generate_connect_family(record, ranker, field_model, geo, *, max_sources=16, per_component=2, max_pairs=48):
    infos=ranked_endpoint_infos(record,ranker,per_component=per_component,max_sources=max_sources)
    pairs=compatible_pairs(infos,max_pairs=max_pairs)
    im=record['image'].astype(np.float32);p=record['prob'].astype(np.float32);base=record['base'].astype(bool);thr=float(record['threshold'])
    out=[]
    for pair in pairs:out.extend(bidirectional_connect_candidates(field_model,geo,im,p,base,thr,pair))
    return out,infos,pairs

@dataclass
class RidgeBeam:
    path:list
    hist_xy:list
    score:float
    min_ridge:float


def _geo_step_vec(geo,hist_xy):
    from .geometry import predict_future
    h=np.asarray(hist_xy,np.float32)
    try:
        p=predict_future(geo,h,1)[0];v=p-h[-1]
    except Exception:
        v=h[-1]-h[-2]
    n=float(np.linalg.norm(v));return (v/(n+1e-8)).astype(np.float32)


def ridge_continue_candidates(record, info:EndpointInfo, geo, *, beam_width=16, branch_k=5, max_steps=72, checkpoint_every=4):
    """Open-ended CONTINUE proposal family based on local RGB ridge + geometry continuity.

    Unlike V5.3's fixed source field corridor, this walker can follow a long curved visible
    ridge. It is proposal-only and deliberately emits multiple prefixes for later ranking.
    """
    image=record['image'].astype(np.float32);base=record['base'].astype(bool);prob=record['prob'].astype(np.float32);thr=float(record['threshold'])
    bh,dark,gr=evidence(image);ridge=np.maximum(bh,dark);rp=relprob(prob,thr);H,W=base.shape
    sy,sx=info.anchor_yx
    b0=RidgeBeam([(sy,sx)],[tuple(map(float,p)) for p in info.history_xy],0.,1.)
    beams=[b0];out=[]
    for step in range(max_steps):
        new=[]
        for b in beams:
            cy,cx=b.path[-1];gv=_geo_step_vec(geo,np.asarray(b.hist_xy[-6:],np.float32))
            prev=np.asarray(b.hist_xy[-1],np.float32)-np.asarray(b.hist_xy[-2],np.float32);prev/=np.linalg.norm(prev)+1e-8
            cand=[]
            for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                y,x=cy+dy,cx+dx
                if not (0<=y<H and 0<=x<W) or base[y,x] or (y,x) in b.path:continue
                mv=np.array([float(dx),float(dy)],np.float32);mv/=np.linalg.norm(mv)+1e-8
                cont=float(np.dot(prev,mv));gcompat=float(np.dot(gv,mv));rv=float(ridge[y,x])
                local=.46*rv+.24*cont+.24*gcompat+.06*float(rp[y,x])
                if cont<-.25:local-=.35
                cand.append((local,y,x,rv,mv))
            cand.sort(reverse=True,key=lambda z:z[0])
            for local,y,x,rv,mv in cand[:branch_k]:
                nh=list(b.hist_xy);nh.append((float(x),float(y)))
                new.append(RidgeBeam(b.path+[(y,x)],nh,b.score+local,min(b.min_ridge,rv)))
        if not new:break
        new=sorted(new,key=lambda z:z.score/max(len(z.path)-1,1),reverse=True)[:beam_width];beams=new
        L=step+1
        if L>=2 and (L%checkpoint_every==0 or L in (6,10,14,18,22,30,40,50,60,70)):
            for b in beams:
                add=np.zeros(base.shape,bool)
                for y,x in b.path[1:]:add[y,x]=1
                vals=ridge[add]
                out.append({'family':'v54_ridge_continue','add':add,'path':add.copy(),'path_yx':b.path,'source_yx':info.anchor_yx,'source_endpoint_yx':info.endpoint_yx,
                            'source_component':info.component_id,'source_score':info.source_score,'score':float(b.score/max(L,1)),'mean_ridge':float(vals.mean()) if vals.size else 0.,
                            'min_ridge':float(vals.min()) if vals.size else 0.,'length':int(add.sum()),'connects_foreign':False,'learned_stop':False})
    seen=set();keep=[]
    for z in sorted(out,key=lambda q:(q['score'],q['mean_ridge']),reverse=True):
        k=np.packbits(z['add']).tobytes()
        if k in seen:continue
        seen.add(k);keep.append(z)
    return keep


def generate_continue_family(record, ranker, geo, *, max_sources=12, per_component=2):
    infos=ranked_endpoint_infos(record,ranker,per_component=per_component,max_sources=max_sources)
    out=[]
    for info in infos:out.extend(ridge_continue_candidates(record,info,geo))
    return out,infos

def _compress_ridge_prefixes(zs, n=48):
    pool=[]
    for key in ('score','mean_ridge','length'):
        pool += sorted(zs,key=lambda z:float(z.get(key,0.)),reverse=True)[:max(8,n//3)]
    seen=set();out=[]
    for z in pool:
        k=np.packbits(z['add']).tobytes()
        if k in seen:continue
        seen.add(k);out.append(z)
    return out[:n]


def _line_pixels(a,b,shape):
    m=np.zeros(shape,np.uint8);cv2.line(m,(int(a[1]),int(a[0])),(int(b[1]),int(b[0])),1,1,lineType=cv2.LINE_8);return m.astype(bool)


def bidirectional_ridge_meet_candidates(record, a:EndpointInfo, b:EndpointInfo, geo, *, join_radius=7, beam_width=8, branch_k=3):
    """Meet-in-the-middle CONNECT: trace locally from both endpoints and join nearby fronts."""
    d=float(np.linalg.norm(np.asarray(a.anchor_yx,float)-np.asarray(b.anchor_yx,float)));steps=int(np.clip(math.ceil(.65*d)+8,8,36))
    za=_compress_ridge_prefixes(ridge_continue_candidates(record,a,geo,beam_width=beam_width,branch_k=branch_k,max_steps=steps,checkpoint_every=2),48)
    zb=_compress_ridge_prefixes(ridge_continue_candidates(record,b,geo,beam_width=beam_width,branch_k=branch_k,max_steps=steps,checkpoint_every=2),48)
    base=record['base'].astype(bool);out=[];pairs=[]
    for pa in za:
        ea=tuple(map(int,pa['path_yx'][-1]))
        for pb in zb:
            eb=tuple(map(int,pb['path_yx'][-1]));gap=float(np.linalg.norm(np.asarray(ea,float)-np.asarray(eb,float)))
            if gap>join_radius:continue
            join=_line_pixels(ea,eb,base.shape)
            if np.any(join&base):continue
            def lastv(z):
                p=z['path_yx'];
                if len(p)<2:return np.array([0.,0.])
                y0,x0=p[-2];y1,x1=p[-1];v=np.array([x1-x0,y1-y0],float);return v/(np.linalg.norm(v)+1e-9)
            va=lastv(pa);vb=lastv(pb);jv=np.array([eb[1]-ea[1],eb[0]-ea[0]],float);jn=np.linalg.norm(jv)
            if jn>1e-6:
                ju=jv/jn
                if np.dot(va,ju)<-.35 or np.dot(vb,-ju)<-.35:continue
            add=(pa['add']|pb['add']|join)&~base
            if int(add.sum())<2:continue
            score=.45*float(pa['score'])+.45*float(pb['score'])-.10*(gap/max(join_radius,1))
            pairs.append((score,gap,pa,pb,add))
    seen=set()
    for score,gap,pa,pb,add in sorted(pairs,key=lambda x:(x[0],-x[1]),reverse=True)[:96]:
        k=np.packbits(add).tobytes()
        if k in seen:continue
        seen.add(k);out.append({'family':'v54_bidir_ridge_meet','add':add,'path':add.copy(),'source_yx':a.anchor_yx,'destination_yx':b.anchor_yx,'source_endpoint_yx':a.endpoint_yx,'destination_endpoint_yx':b.endpoint_yx,
          'source_component':a.component_id,'destination_component':b.component_id,'source_score':a.source_score,'destination_score':b.source_score,'pair_distance':d,'meet_gap':gap,'score':float(score),'mean_ridge':.5*(float(pa['mean_ridge'])+float(pb['mean_ridge'])),'min_ridge':min(float(pa['min_ridge']),float(pb['min_ridge'])),'length':int(add.sum()),'connects_foreign':True,'learned_stop':True,
          'path_a_yx':pa['path_yx'],'path_b_yx':pb['path_yx']})
    return out


def generate_bidir_meet_family(record, ranker, geo, *, max_sources=12, per_component=2, max_pairs=24):
    infos=ranked_endpoint_infos(record,ranker,per_component=per_component,max_sources=max_sources)
    pairs=compatible_pairs(infos,min_distance=3,max_distance=48,facing_min=-.65,facing_sum_min=-.75,max_pairs=max_pairs)
    out=[]
    for q in pairs:out.extend(bidirectional_ridge_meet_candidates(record,q['a'],q['b'],geo))
    return out,infos,pairs
