from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from lightgbm import LGBMClassifier
from .actions import action_targets
from .features import build_features

@dataclass(frozen=True)
class TrainingConfig:
    add_margin: float=.20
    add_radial_max: float=2.5
    add_max_positive: int=700
    add_max_negative: int=1400
    remove_max_keep: int=1400
    remove_max_positive: int=450
    remove_min_distance_ratio: float=1.25
    hard_keep_fraction: float=.70
    n_estimators: int=180
    learning_rate: float=.04
    num_leaves: int=15
    max_depth: int=5
    min_child_samples: int=30
    reg_lambda: float=.5

def stable_seed(seed,*parts):
    h=hashlib.sha256("|".join([str(int(seed)),*map(str,parts)]).encode()).digest(); return int(int.from_bytes(h[:8],"big")%(2**31-1))

def _sample(idx,n,rng):
    idx=np.asarray(idx,np.int64)
    if n<=0 or not len(idx):return np.empty(0,np.int64)
    return idx if len(idx)<=n else np.asarray(rng.choice(idx,n,replace=False),np.int64)

def add_candidate(probability,base_mask,features,names,base_threshold,config:TrainingConfig):
    p=np.asarray(probability,np.float32); b=np.asarray(base_mask,bool); radial=features[...,names.index("radial_ratio")]
    if not b.any():return np.zeros_like(b)
    return (~b)&(p>=max(.01,float(base_threshold)-float(config.add_margin)))&(radial>0)&(radial<=float(config.add_radial_max))


def _safe_remove(base,gt,min_ratio):
    fp=base&~gt
    if not fp.any():return fp
    if not gt.any():return fp
    sk=skeletonize(gt); dist=ndi.distance_transform_edt(~gt).astype(np.float32)
    if sk.any():
        _,idx=ndi.distance_transform_edt(~sk,return_indices=True); rad=ndi.distance_transform_edt(gt).astype(np.float32); ratio=dist/np.maximum(rad[idx[0],idx[1]],1.)
    else:ratio=dist
    lab,n=ndi.label(fp); out=np.zeros_like(fp)
    for i in range(1,n+1):
        r=lab==i
        if r.any() and float(np.min(ratio[r]))>=float(min_ratio):out|=r
    return out

def _hard_keep(gt,keep):
    diag=max(float(np.hypot(*gt.shape)),1.); r=max(1,int(round(.006*diag))); er=ndi.binary_erosion(gt,iterations=r,border_value=0); return keep&(gt&~er)

def _validate(records,probabilities,base_threshold,c):
    if not np.isfinite(base_threshold) or not 0<=float(base_threshold)<=1:raise ValueError("bad base threshold")
    if c.add_margin<0 or c.add_radial_max<=0 or c.remove_min_distance_ratio<0 or not 0<=c.hard_keep_fraction<=1:raise ValueError("bad training config")
    if min(c.add_max_positive,c.add_max_negative,c.remove_max_keep,c.remove_max_positive)<1:raise ValueError("sample limits must be positive")
    if c.n_estimators<1 or c.learning_rate<=0 or c.num_leaves<2 or c.max_depth<1 or c.min_child_samples<1 or c.reg_lambda<0:raise ValueError("bad learner config")
    recs=sorted(list(records),key=lambda r:str(r.get("name","")).strip()); names=[str(r.get("name","")).strip() for r in recs]
    if not names or any(not x for x in names):raise ValueError("stable record names required")
    if len(set(names))!=len(names):raise ValueError("duplicate record names")
    if any(n not in probabilities for n in names):raise KeyError("missing probabilities")
    return recs

def build_training_matrices(records,probabilities,base_threshold,seed=1337,config:TrainingConfig|None=None):
    """Build ADD and KEEP/REMOVE rows from natural frozen-Base errors only.

    Each record must contain an image, a pixel GT mask and a matching frozen-Base
    probability map. Coordinate-only structural trajectories are intentionally not
    accepted by this API. CRCV generates no crack and no synthetic error internally.
    """
    c=config or TrainingConfig(); recs=_validate(records,probabilities,base_threshold,c); AX=[];Ay=[];RX=[];Ry=[];schema=None; source_counts={}
    for r in recs:
        n=str(r["name"]).strip(); im=np.asarray(r["image"],np.float32); gt=np.asarray(r["gt"],bool); p=np.asarray(probabilities[n],np.float32)
        if im.ndim!=3 or gt.ndim!=2 or im.shape[:2]!=gt.shape or p.shape!=gt.shape or not np.isfinite(im).all() or not np.isfinite(p).all():raise ValueError(f"bad record {n}")
        src=str(r.get("source","unspecified")); source_counts[src]=source_counts.get(src,0)+1; b=p>=float(base_threshold); X,names=build_features(im,p,b); schema=schema or names
        if schema!=names:raise AssertionError("feature schema drift")
        t=action_targets(b,gt); rng=np.random.default_rng(stable_seed(seed,n,"add")); cand=add_candidate(p,b,X,names,base_threshold,c); pi=_sample(np.flatnonzero((cand&t["add"]).ravel()),c.add_max_positive,rng); ni=_sample(np.flatnonzero((cand&~gt).ravel()),c.add_max_negative,rng); idx=np.r_[pi,ni]
        if len(idx):AX.append(X.reshape(-1,X.shape[-1])[idx]); Ay.append(np.r_[np.ones(len(pi),np.int8),np.zeros(len(ni),np.int8)])
        keep=t["keep"]; rem=_safe_remove(b,gt,c.remove_min_distance_ratio); hard=_hard_keep(gt,keep); easy=keep&~hard; rng=np.random.default_rng(stable_seed(seed,n,"remove")); hi=_sample(np.flatnonzero(hard.ravel()),int(round(c.remove_max_keep*c.hard_keep_fraction)),rng); ei=_sample(np.flatnonzero(easy.ravel()),c.remove_max_keep-len(hi),rng); ri=_sample(np.flatnonzero(rem.ravel()),c.remove_max_positive,rng); ridx=np.r_[np.tile(hi,3),ei,ri]
        if len(ridx):RX.append(X.reshape(-1,X.shape[-1])[ridx]); Ry.append(np.r_[np.zeros(len(hi)*3+len(ei),np.int8),np.ones(len(ri),np.int8)])
    if not AX or not RX:raise ValueError("insufficient ADD/REMOVE training rows")
    ax=np.concatenate(AX).astype(np.float32); ay=np.concatenate(Ay).astype(np.int8); rx=np.concatenate(RX).astype(np.float32); ry=np.concatenate(Ry).astype(np.int8)
    if len(np.unique(ay))<2 or len(np.unique(ry))<2:raise ValueError("both action heads require two classes")
    schema_sha=hashlib.sha256(json.dumps(schema,separators=(",",":")).encode()).hexdigest(); matrix_sha=hashlib.sha256(ax.tobytes()+ay.tobytes()+rx.tobytes()+ry.tobytes()).hexdigest()
    meta={"method":"CRCV-V5.21","core_version":"1.1.1","feature_names":schema,"feature_schema_sha256":schema_sha,"training_matrix_sha256":matrix_sha,"sources":source_counts,"add_rows":int(len(ay)),"add_positive":int(ay.sum()),"remove_rows":int(len(ry)),"remove_positive":int(ry.sum())}
    return (ax,ay),(rx,ry),meta


def select_asymmetric_operating_point(rows,target_gain:float=.01):
    """Select CAL thresholds with asymmetric action risk.

    Rows are ``(min_gain, dice_gain, add_tau, remove_tau, payload)``.
    If the requested gain is attainable, preserve the most conservative
    REMOVE threshold and optimize recovery within that REMOVE level instead
    of also maximizing the ADD threshold. If not attainable, fall back to
    the best balanced CAL point.
    """
    rows=list(rows)
    if not rows: raise ValueError("no CAL operating points")
    target=[x for x in rows if float(x[0])>=float(target_gain)]
    if not target: return max(rows,key=lambda x:(x[0],x[1]))
    remove_tau=max(float(x[3]) for x in target)
    same=[x for x in target if float(x[3])==remove_tau]
    return max(same,key=lambda x:(x[0],x[1],-x[2]))

def _fit(X,y,c,seed):
    return LGBMClassifier(n_estimators=c.n_estimators,learning_rate=c.learning_rate,num_leaves=c.num_leaves,max_depth=c.max_depth,min_child_samples=c.min_child_samples,reg_lambda=c.reg_lambda,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed,data_random_seed=seed,deterministic=True,force_col_wise=True,n_jobs=1,verbosity=-1).fit(X,y)

def train(records,probabilities,base_threshold,seed=1337,config:TrainingConfig|None=None):
    c=config or TrainingConfig(); (ax,ay),(rx,ry),meta=build_training_matrices(records,probabilities,base_threshold,seed,c); add=_fit(ax,ay,c,stable_seed(seed,"ADD")); remove=_fit(rx,ry,c,stable_seed(seed,"REMOVE")); meta.update({"seed":int(seed),"base_threshold":float(base_threshold),"training_config":asdict(c)}); return {"add":add,"remove":remove},meta
