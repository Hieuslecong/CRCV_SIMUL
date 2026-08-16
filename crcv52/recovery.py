from __future__ import annotations
import heapq, cv2, numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .geometry import predict_future
N8=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
def degree(sk):
    k=np.ones((3,3),np.uint8);k[1,1]=0;return cv2.filter2D(sk.astype(np.uint8),-1,k,borderType=cv2.BORDER_CONSTANT)
def endpoints(mask):
    sk=skeletonize(mask.astype(bool));d=degree(sk);return [tuple(map(int,x)) for x in np.argwhere(sk&(d==1))]
def trace_inward(mask,ep,n=6):
    sk=skeletonize(mask.astype(bool));dist=distance_transform_edt(mask);cur=tuple(map(int,ep));prev=None;path=[cur];prev_vec=None
    for _ in range(n-1):
        y,x=cur;ns=[]
        for dy,dx in N8:
            yy,xx=y+dy,x+dx
            if 0<=yy<sk.shape[0] and 0<=xx<sk.shape[1] and sk[yy,xx] and (yy,xx)!=prev:ns.append((yy,xx))
        if not ns:break
        if prev_vec is None: nxt=max(ns,key=lambda z:dist[z])
        else:
            pv=np.asarray(prev_vec,float);pn=np.linalg.norm(pv)+1e-8
            def score(z):
                v=np.array([z[0]-y,z[1]-x],float);return .9*float(np.dot(pv,v)/(pn*(np.linalg.norm(v)+1e-8)))+.1*float(dist[z]/(dist.max()+1e-8))
            nxt=max(ns,key=score)
        prev_vec=(cur[0]-nxt[0],cur[1]-nxt[1]);prev,cur=cur,nxt;path.append(cur)
    if len(path)<n:return None
    return np.asarray([(x,y) for y,x in path[::-1]],np.float32)
def poly_mask(path,shape):
    m=np.zeros(shape,np.uint8);pts=np.asarray([(int(round(x)),int(round(y))) for x,y in path],np.int32)
    if len(pts)>=2:cv2.polylines(m,[pts],False,1,1,lineType=cv2.LINE_8)
    return m.astype(bool)
def corridor_from_prior(prior,shape,r=6):
    c=poly_mask(prior,shape).astype(np.uint8);return cv2.dilate(c,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)),1).astype(bool)
def rotate(path,o,deg):
    a=np.deg2rad(deg);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]],np.float32);return (path-o)@R.T+o
def route(cost,allowed,start,target):
    H,W=allowed.shape;sy,sx=start;ty,tx=target
    if not (0<=sy<H and 0<=sx<W and 0<=ty<H and 0<=tx<W):return None
    if not allowed[sy,sx] or not allowed[ty,tx]:return None
    D=np.full((H,W),np.inf,np.float64);D[sy,sx]=0;P=np.full((H,W,2),-1,np.int16);hq=[(0.,sy,sx)]
    while hq:
        d,y,x=heapq.heappop(hq)
        if d>D[y,x]+1e-12:continue
        if (y,x)==(ty,tx):break
        for dy,dx in N8:
            yy,xx=y+dy,x+dx
            if 0<=yy<H and 0<=xx<W and allowed[yy,xx]:
                nd=d+(1.414 if dy and dx else 1.)*(.01+float(cost[yy,xx]))
                if nd<D[yy,xx]:D[yy,xx]=nd;P[yy,xx]=(y,x);heapq.heappush(hq,(nd,yy,xx))
    if not np.isfinite(D[ty,tx]):return None
    cur=(ty,tx);path=[]
    for _ in range(H*W):
        path.append(cur)
        if cur==(sy,sx):break
        py,px=P[cur]
        if py<0:return None
        cur=(int(py),int(px))
    return path[::-1]
def source_proposals(geo,base,max_sources=8,horizons=(6,12,18,24),angles=(-8,0,8),corridor_radius=6,endpoint_subset=None):
    E=list(endpoint_subset) if endpoint_subset is not None else endpoints(base);out=[]
    for ep in E[:max_sources*2]:
        hist=trace_inward(base,ep,6)
        if hist is None:continue
        fut=predict_future(geo,hist,max(horizons))
        for h in horizons:
            for a in angles:
                pr=rotate(fut[:h],hist[-1],a);full=np.vstack([hist[-1],pr]);corr=corridor_from_prior(full,base.shape,corridor_radius);out.append({'source_yx':ep,'history':hist,'prior':pr,'corridor':corr,'horizon':h,'angle':a})
    return out

def skeleton_paths(mask,min_len=12):
    sk=skeletonize(mask.astype(bool));deg=degree(sk);seen=set();paths=[]
    for start in map(tuple,np.argwhere(sk&(deg==1))):
        if start in seen:continue
        path=[start];prev=None;cur=start
        for _ in range(sk.size):
            seen.add(cur);y,x=cur;ns=[]
            for dy,dx in N8:
                q=(y+dy,x+dx)
                if 0<=q[0]<sk.shape[0] and 0<=q[1]<sk.shape[1] and sk[q] and q!=prev:ns.append(q)
            if not ns:break
            cand=[q for q in ns if q not in seen] or ns
            if prev is not None:
                pv=np.array([cur[0]-prev[0],cur[1]-prev[1]],float)
                def sc(q):
                    v=np.array([q[0]-cur[0],q[1]-cur[1]],float);return np.dot(pv,v)/(np.linalg.norm(pv)*np.linalg.norm(v)+1e-9)
                nxt=max(cand,key=sc)
            else:nxt=cand[0]
            prev,cur=cur,nxt;path.append(cur)
            if deg[cur]!=2:break
        if len(path)>=min_len:paths.append(path)
    return paths

def boundary_exit_anchor(mask,ep,history_xy,max_radius=12):
    b=mask.astype(bool);hist=np.asarray(history_xy,np.float32);y,x=map(int,ep)
    if len(hist)<2:return (y,x),hist
    v=hist[-1]-hist[-2];v=v/(np.linalg.norm(v)+1e-8)
    H,W=b.shape
    bg=(~b).astype(np.uint8);boundary=b & (cv2.dilate(bg,np.ones((3,3),np.uint8),1).astype(bool))
    ys,xs=np.where(boundary)
    best=None
    for yy,xx in zip(ys,xs):
        dx=float(xx-x);dy=float(yy-y);d=np.hypot(dx,dy)
        if d<.5 or d>max_radius:continue
        proj=dx*v[0]+dy*v[1]
        if proj<=0:continue
        lateral=abs(dx*v[1]-dy*v[0])
        score=proj-1.5*lateral-.05*d
        if best is None or score>best[0]:best=(score,(int(yy),int(xx)))
    if best is None:return (y,x),hist
    ay,ax=best[1]
    if (ay,ax)==(y,x):return (y,x),hist
    ext=np.vstack([hist,np.array([[ax,ay]],np.float32)])
    return (ay,ax),ext
