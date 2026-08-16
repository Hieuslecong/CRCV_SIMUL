from __future__ import annotations
from dataclasses import dataclass
import math, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from .recovery import N8, corridor_from_prior, boundary_exit_anchor
from .geometry import predict_future
from .field import relprob, crop_spec, place, PATCH
from .field_b import candidate_tensor_b, evidence

DIRS=list(N8)
STOP_CLASS=8
PATCH_TR=21
R=PATCH_TR//2

class IterativeTracerNet(nn.Module):
    """Tiny next-node + STOP + same-crack identity head."""
    def __init__(self, cin=12, c=20):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(cin,c,3,1,1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,1,1,groups=c,bias=False),nn.Conv2d(c,c,1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,2,1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,1,1,groups=c,bias=False),nn.Conv2d(c,24,1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),nn.AdaptiveAvgPool2d((3,3)))
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(24*9,64),nn.SiLU());self.next=nn.Linear(64,9);self.identity=nn.Linear(64,1)
    def forward(self,x):
        h=self.fc(self.body(x));return self.next(h),self.identity(h)[:,0]

def _patch(a,cy,cx,size=PATCH_TR):
    r=size//2;aa=a[...,None] if a.ndim==2 else a;out=np.zeros((size,size,aa.shape[2]),aa.dtype);y0=max(0,cy-r);y1=min(aa.shape[0],cy+r+1);x0=max(0,cx-r);x1=min(aa.shape[1],cx+r+1);oy0=y0-(cy-r);ox0=x0-(cx-r);out[oy0:oy0+y1-y0,ox0:ox0+x1-x0]=aa[y0:y1,x0:x1];return out[...,0] if a.ndim==2 else out

def source_field_map(field_model,geo,image,prob,base,thr,source_yx,history_xy,horizon=24,corridor_radius=6):
    fut=predict_future(geo,history_xy,horizon);o=np.asarray(history_xy[-1],np.float32);corr=np.zeros(base.shape,bool)
    for deg in (-8.,0.,8.):
        a=np.deg2rad(deg);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]],np.float32);pr=(fut-o)@R.T+o;corr|=corridor_from_prior(np.vstack([o,pr]),base.shape,corridor_radius)
    q={'source_yx':tuple(map(int,source_yx)),'history':history_xy,'prior':fut,'corridor':corr,'horizon':horizon,'angle':0.};X,(y0,x0)=candidate_tensor_b(image,prob,base,thr,q,None)
    with torch.no_grad(): f=torch.sigmoid(field_model(torch.tensor(X[None],dtype=torch.float32)))[0].numpy()
    full=place(f,base.shape,y0,x0).astype(np.float32);return full,q

def source_field_atlas(field_model,geo,image,prob,base,thr,source_yx,history_xy,total_horizon=64,local_horizon=24,stride=12,corridor_radius=6,max_corridor_radius=16):
    hist=np.asarray(history_xy,np.float32);future=predict_future(geo,hist,total_horizon);allpts=np.vstack([hist,future]);field=np.zeros(base.shape,np.float32);corr=np.zeros(base.shape,bool);segments=[];starts=list(range(0,max(total_horizon,1),max(int(stride),1)))
    for st in starts:
        if st==0:h=hist;cur_yx=tuple(map(int,source_yx))
        else:
            idx=len(hist)+st-1;lo=max(0,idx-5);h=allpts[lo:idx+1]
            if len(h)<2:continue
            x,y=h[-1];cur_yx=(int(round(y)),int(round(x)))
            if not (0<=cur_yx[0]<base.shape[0] and 0<=cur_yx[1]<base.shape[1]):continue
        try:rr=int(min(max_corridor_radius, corridor_radius + round(0.18*st)));fm,q=source_field_map(field_model,geo,image,prob,base,thr,cur_yx,h,local_horizon,rr)
        except Exception:continue
        field=np.maximum(field,fm);corr|=q['corridor'];segments.append({'start_step':int(st),'source_yx':cur_yx,'radius':int(rr),'corridor':q['corridor']})
        if st+local_horizon>=total_horizon:break
    return field,{'corridor':corr,'prior':future,'segments':segments,'source_yx':tuple(map(int,source_yx))}

def _geom_vec(geo,hist_xy):
    try:p=predict_future(geo,np.asarray(hist_xy,np.float32),1)[0];v=p-np.asarray(hist_xy[-1],np.float32)
    except Exception:v=np.asarray(hist_xy[-1])-np.asarray(hist_xy[-2])
    n=float(np.linalg.norm(v));return (v/(n+1e-8)).astype(np.float32)

def prepare_image_maps(image,prob,base,thr):
    bh,dark,gr=evidence(image);rp=relprob(prob,thr);return {'rp':rp.astype(np.float32),'bh':bh.astype(np.float32),'dark':dark.astype(np.float32),'gr':gr.astype(np.float32)}
def state_tensor(image,prob,base,thr,current_yx,hist_xy,field_map,geo,maps=None):
    cy,cx=map(int,current_yx);maps=prepare_image_maps(image,prob,base,thr) if maps is None else maps;bh,dark,gr,rp=maps['bh'],maps['dark'],maps['gr'],maps['rp'];hist=np.zeros(base.shape,np.float32);pts=np.rint(np.asarray(hist_xy,np.float32)).astype(np.int32)
    if len(pts)>=2:cv2.polylines(hist,[pts],False,1.,1,lineType=cv2.LINE_8)
    elif len(pts)==1:
        x,y=pts[0]
        if 0<=y<hist.shape[0] and 0<=x<hist.shape[1]:hist[y,x]=1
    gv=_geom_vec(geo,hist_xy);im=_patch(image,cy,cx);chans=[im[...,0],im[...,1],im[...,2],_patch(rp,cy,cx),_patch(base.astype(np.float32),cy,cx),_patch(bh,cy,cx),_patch(dark,cy,cx),_patch(gr,cy,cx),_patch(field_map,cy,cx),_patch(hist,cy,cx),np.full((PATCH_TR,PATCH_TR),gv[0],np.float32),np.full((PATCH_TR,PATCH_TR),gv[1],np.float32)];return np.stack(chans).astype(np.float32)
def dir_class(cur,nxt):
    dy=int(nxt[0]-cur[0]);dx=int(nxt[1]-cur[1]);return DIRS.index((dy,dx)) if (dy,dx) in DIRS else None

@dataclass
class Beam:
    path:list;hist_xy:list;score:float;min_identity:float;stopped:bool=False;stop_prob:float=0.;connects_foreign:bool=False

def _adjacent_foreign(base_labels,source_cid,y,x):
    H,W=base_labels.shape
    for dy,dx in N8:
        yy,xx=y+dy,x+dx
        if 0<=yy<H and 0<=xx<W:
            cid=int(base_labels[yy,xx])
            if cid>0 and cid!=source_cid:return True
    return False

def ranked_valid_moves(log_probs, current_yx, base, labels, source_cid, corridor, path, branch_k):
    cur=tuple(map(int,current_yx));ranked=np.argsort(np.asarray(log_probs))[::-1];out=[]
    for a in ranked:
        a=int(a)
        if a==STOP_CLASS:continue
        dy,dx=DIRS[a];y,x=cur[0]+dy,cur[1]+dx
        if not (0<=y<base.shape[0] and 0<=x<base.shape[1]):continue
        if not corridor[y,x] or (y,x) in path:continue
        cid=int(labels[y,x])
        if cid>0:continue
        out.append((a,y,x))
        if len(out)>=branch_k:break
    return out

def trace_source(model,field_model,geo,image,prob,base,thr,source_yx,history_xy,beam_width=4,branch_k=3,max_steps=32,min_steps=4,stop_bias=0.,identity_floor=0.10,rolling_horizon=None,soft_corridor_after=None):
    source_endpoint=tuple(map(int,source_yx));source_yx,history_xy=boundary_exit_anchor(base,source_endpoint,history_xy,12)
    if rolling_horizon is not None and int(rolling_horizon)>24:field,q=source_field_atlas(field_model,geo,image,prob,base,thr,source_yx,history_xy,int(rolling_horizon),24,12,6)
    else:field,q=source_field_map(field_model,geo,image,prob,base,thr,source_yx,history_xy,24,6)
    corr=q['corridor'];n,labels=cv2.connectedComponents(base.astype(np.uint8),8);sy,sx=map(int,source_yx);source_cid=int(labels[sy,sx]);maps=prepare_image_maps(image,prob,base,thr);b0=Beam(path=[(sy,sx)],hist_xy=[tuple(map(float,p)) for p in history_xy],score=0.,min_identity=1.);beams=[b0];completed=[]
    for step in range(max_steps):
        active=[b for b in beams if not b.stopped]
        if not active:break
        X=np.stack([state_tensor(image,prob,base,thr,b.path[-1],np.asarray(b.hist_xy[-6:],np.float32),field,geo,maps) for b in active])
        with torch.no_grad():logits,ilog=model(torch.tensor(X,dtype=torch.float32));lp=F.log_softmax(logits,1).numpy();ident=torch.sigmoid(ilog).numpy()
        new=[]
        for b,lpi,ii in zip(active,lp,ident):
            cur=b.path[-1];stop_p=float(np.exp(lpi[STOP_CLASS]));acts=np.argsort(lpi)[::-1]
            if len(b.path)-1>=min_steps:completed.append(Beam(list(b.path),list(b.hist_xy),b.score+float(lpi[STOP_CLASS])+stop_bias,min(b.min_identity,float(ii)),True,stop_p,b.connects_foreign))
            valid_moves=0
            for a in acts:
                if a==STOP_CLASS:continue
                dy,dx=DIRS[int(a)];y,x=cur[0]+dy,cur[1]+dx
                if not(0<=y<base.shape[0] and 0<=x<base.shape[1]):continue
                inside_corr=bool(corr[y,x])
                if (not inside_corr) and (soft_corridor_after is None or step < int(soft_corridor_after)):continue
                if (y,x) in b.path:continue
                cid=int(labels[y,x])
                if cid>0:
                    if cid!=source_cid and _adjacent_foreign(labels,source_cid,cur[0],cur[1]):completed.append(Beam(list(b.path),list(b.hist_xy),b.score+float(lpi[a]),min(b.min_identity,float(ii)),True,stop_p,True))
                    continue
                nh=list(b.hist_xy);nh.append((float(x),float(y)));gv=_geom_vec(geo,np.asarray(b.hist_xy[-6:],np.float32));mv=np.array([float(dx),float(dy)],np.float32);mv/=np.linalg.norm(mv)+1e-8;gcompat=float(np.dot(gv,mv));corr_term=.08 if inside_corr else -.18;sc=b.score+float(lpi[a])+.35*math.log(max(float(ii),1e-6))+.15*math.log(max(float(field[y,x]),1e-5))+.30*gcompat+corr_term;nb=Beam(b.path+[(y,x)],nh,sc,min(b.min_identity,float(ii)),False,stop_p,b.connects_foreign)
                if nb.min_identity>=identity_floor:
                    new.append(nb);valid_moves+=1
                    if valid_moves>=branch_k:break
        if not new:break
        new=sorted(new,key=lambda z:z.score/max(len(z.path)-1,1),reverse=True)[:beam_width];beams=new;step_len=step+1
        if step_len in tuple(range(4,max_steps+1,4)):completed += [Beam(list(b.path),list(b.hist_xy),b.score,b.min_identity,False,b.stop_prob,b.connects_foreign) for b in beams if len(b.path)-1>=min_steps]
    completed += [Beam(list(b.path),list(b.hist_xy),b.score,b.min_identity,True,b.stop_prob,b.connects_foreign) for b in beams if len(b.path)-1>=min_steps]
    out=[];seen=set()
    for b in sorted(completed,key=lambda z:z.score/max(len(z.path)-1,1),reverse=True):
        pts=b.path[1:]
        if len(pts)<min_steps:continue
        add=np.zeros(base.shape,bool)
        for y,x in pts:add[y,x]=1
        key=np.packbits(add).tobytes()
        if key in seen:continue
        seen.add(key);fvals=np.asarray([field[y,x] for y,x in pts],np.float32);score=b.score/max(len(pts),1);out.append({'add':add,'path_yx':pts,'score':float(score),'min_identity':float(b.min_identity),'stop_prob':float(b.stop_prob),'mean_field':float(fvals.mean()),'min_field':float(fvals.min()),'length':len(pts),'connects_foreign':bool(b.connects_foreign),'learned_stop':bool(b.stopped),'source_yx':(sy,sx),'source_endpoint_yx':source_endpoint,'corridor':corr})
    return out,field,q
