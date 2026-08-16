from __future__ import annotations
import cv2, numpy as np
from skimage.morphology import skeletonize
from .recovery import endpoints, trace_inward, boundary_exit_anchor, N8

def _neighbors(mask,p):
    y,x=p;out=[]
    for dy,dx in N8:
        yy,xx=y+dy,x+dx
        if 0<=yy<mask.shape[0] and 0<=xx<mask.shape[1] and mask[yy,xx]:out.append((yy,xx))
    return out

def _nearest_missing_seed(missing,ep,history_xy=None,radius=8):
    y,x=ep;best=None
    outv=None
    if history_xy is not None and len(history_xy)>=2:
        v=np.asarray(history_xy[-1]-history_xy[-2],float) # xy outward
        outv=v/(np.linalg.norm(v)+1e-9)
    for yy in range(max(0,y-radius),min(missing.shape[0],y+radius+1)):
        for xx in range(max(0,x-radius),min(missing.shape[1],x+radius+1)):
            if not missing[yy,xx]:continue
            vx,vy=float(xx-x),float(yy-y);d=np.hypot(vx,vy)
            if d<.5 or d>radius:continue
            align=1.
            if outv is not None:
                vv=np.array([vx,vy]);align=float(np.dot(outv,vv)/(np.linalg.norm(vv)+1e-9))
                if align<-.10:continue
            # proximity first, then forward alignment
            score=d-.75*align
            if best is None or score<best[0]:best=(score,(yy,xx))
    return None if best is None else best[1]

def trace_missing_from_endpoint(gt,base,ep,max_len=40,seed_radius=8,border_margin=8):
    H,W=base.shape;y,x=map(int,ep)
    if y<border_margin or x<border_margin or y>=H-border_margin or x>=W-border_margin:return []
    hist=trace_inward(base,ep,6)
    if hist is None:return []
    anchor,hist=boundary_exit_anchor(base,ep,hist,12)
    gsk=skeletonize(gt.astype(bool));missing=gsk&~base.astype(bool)
    seed=_nearest_missing_seed(missing,anchor,hist,seed_radius)
    if seed is None:return []
    path=[seed];prev=anchor;cur=seed
    # initialize continuity from boundary anchor -> missing seed
    prev_vec=np.array([seed[0]-anchor[0],seed[1]-anchor[1]],float)
    for _ in range(max_len-1):
        ns=[q for q in _neighbors(gsk,cur) if q!=prev]
        missing_ns=[q for q in ns if missing[q]]
        if not missing_ns:break
        if len(missing_ns)>1:
            pv=prev_vec;pn=np.linalg.norm(pv)+1e-9
            def sc(q):
                v=np.array([q[0]-cur[0],q[1]-cur[1]],float)
                return float(np.dot(pv,v)/(pn*(np.linalg.norm(v)+1e-9)))
            nxt=max(missing_ns,key=sc)
        else:nxt=missing_ns[0]
        prev_vec=np.array([nxt[0]-cur[0],nxt[1]-cur[1]],float);prev,cur=cur,nxt;path.append(cur)
        # stop at true missing-skeleton branch ambiguity
        if len([q for q in _neighbors(missing,cur) if q!=prev])>1:break
    return path

def natural_events(records,min_len=2,max_len=40,seed_radius=8,border_margin=8):
    events=[]
    for r in records:
        if r.get('typ')!='crack':continue
        for ep in endpoints(r['base']):
            path=trace_missing_from_endpoint(r['gt'],r['base'],ep,max_len=max_len,seed_radius=seed_radius,border_margin=border_margin)
            if len(path)>=min_len:
                hist=trace_inward(r['base'],ep,6)
                if hist is not None:
                    anchor,hist2=boundary_exit_anchor(r['base'],ep,hist,12)
                    events.append({'name':r['name'],'lineage':r['lineage'],'source_endpoint_yx':ep,'source_yx':anchor,'history_xy':hist2,'path_yx':path,'record':r})
    return events
