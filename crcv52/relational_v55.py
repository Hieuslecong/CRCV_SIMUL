from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize

from .field_b import evidence


FAMILIES = ("v52b", "iterative", "ridge_continue", "other")


def _as_float_image(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if float(np.nanmax(x)) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _crop_center(stack: np.ndarray, center_yx: tuple[int, int], size: int) -> np.ndarray:
    if size < 9 or size % 2 == 0:
        raise ValueError("crop size must be odd and >=9")
    r = size // 2
    y, x = map(int, center_yx)
    p = np.pad(stack, ((r, r), (r, r), (0, 0)), mode="reflect")
    yp, xp = y + r, x + r
    return p[yp-r:yp+r+1, xp-r:xp+r+1].copy()


def _circle(shape: tuple[int, int], yx: tuple[int, int], radius: int = 2) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    y, x = map(int, yx)
    if 0 <= y < shape[0] and 0 <= x < shape[1]:
        cv2.circle(m, (x, y), int(radius), 1, -1)
    return m.astype(np.float32)


def _ordered_add_path(add: np.ndarray, source_yx: tuple[int, int]) -> list[tuple[int, int]]:
    """Recover the longest candidate skeleton route starting nearest the source."""
    sk = skeletonize(np.asarray(add, bool))
    pts = np.argwhere(sk)
    if not len(pts):
        return []
    sy, sx = map(int, source_yx)
    d2 = (pts[:, 0] - sy) ** 2 + (pts[:, 1] - sx) ** 2
    start = tuple(map(int, pts[int(np.argmin(d2))]))
    point_set = {tuple(map(int, p)) for p in pts}
    q = deque([start]); parent = {start: None}; dist = {start: 0}; far = start
    while q:
        y, x = q.popleft()
        if dist[(y, x)] > dist[far]:
            far = (y, x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in point_set and nb not in parent:
                    parent[nb] = (y, x); dist[nb] = dist[(y, x)] + 1; q.append(nb)
    path = []
    cur = far
    while cur is not None:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path


def _candidate_end(candidate: dict, add: np.ndarray) -> tuple[int, int]:
    for key in ("destination_yx", "destination_endpoint_yx"):
        if key in candidate:
            y, x = candidate[key]; return int(y), int(x)
    p = candidate.get("path_yx")
    if not p:
        source = tuple(map(int, candidate.get("source_yx", candidate.get("source_endpoint_yx", (0, 0)))))
        p = _ordered_add_path(add, source)
    if p:
        y, x = p[-1]; return int(y), int(x)
    y, x = candidate.get("source_yx", (0, 0)); return int(y), int(x)


def _path_mid(candidate: dict, add: np.ndarray) -> tuple[int, int]:
    p = candidate.get("path_yx")
    if not p:
        source = tuple(map(int, candidate.get("source_yx", candidate.get("source_endpoint_yx", (0, 0)))))
        p = _ordered_add_path(add, source)
    if p:
        y, x = p[len(p)//2]; return int(y), int(x)
    y, x = candidate.get("source_yx", (0, 0)); return int(y), int(x)


def _family_name(candidate: dict) -> str:
    s = str(candidate.get("family", "")).lower()
    if "v52" in s: return "v52b"
    if "iter" in s or "v53" in s: return "iterative"
    if "ridge" in s: return "ridge_continue"
    return "other"


def _mean_on(a: np.ndarray, m: np.ndarray) -> float:
    v = np.asarray(a)[np.asarray(m, bool)]
    return float(v.mean()) if len(v) else 0.0


def build_relation_views(record: dict, candidate: dict, *, sim_score: float = 0.5, crop_size: int = 33):
    """Build GT-free source/path/destination evidence tensors.

    Channels are RGB(3), Base probability, Base mask, candidate mask, RGB ridge,
    and a role mask. Simulation remains a scalar meta feature.
    """
    image = _as_float_image(record["image"])
    prob = np.asarray(record["prob"], np.float32)
    base = np.asarray(record["base"], bool)
    add = np.asarray(candidate["add"], bool)
    if prob.shape != base.shape or add.shape != base.shape or image.shape[:2] != base.shape:
        raise ValueError("image/prob/base/candidate shapes do not match")
    ridge_maps = evidence(image)
    ridge = np.maximum(ridge_maps[0], ridge_maps[1]).astype(np.float32)
    source = tuple(map(int, candidate.get("source_yx", candidate.get("source_endpoint_yx", (0, 0)))))
    dest = _candidate_end(candidate, add); mid = _path_mid(candidate, add)
    common = np.dstack([image, np.clip(prob,0,1), base.astype(np.float32), add.astype(np.float32), np.clip(ridge,0,1)]).astype(np.float32)
    if common.shape[2] != 7:
        raise AssertionError("unexpected channel count")

    def view(center, role):
        z = np.dstack([common, role]).astype(np.float32)
        c = _crop_center(z, center, crop_size)
        return torch.from_numpy(c.transpose(2,0,1)).float()

    source_view = view(source, _circle(base.shape, source, 2))
    path_view = view(mid, add.astype(np.float32))
    dest_view = view(dest, _circle(base.shape, dest, 2))
    length = float(candidate.get("length", int(add.sum())))
    source_score = float(candidate.get("source_score", 0.0))
    raw_score = float(candidate.get("score", candidate.get("pair_score", 0.0)))
    mean_ridge = float(candidate.get("mean_ridge", _mean_on(ridge, add)))
    min_ridge = float(candidate.get("min_ridge", float(ridge[add].min()) if add.any() else 0.0))
    connects_foreign = float(bool(candidate.get("connects_foreign", False)))
    mean_field = float(candidate.get("mean_field", candidate.get("mean_pair_field", 0.0)))
    min_field = float(candidate.get("min_field", candidate.get("min_pair_field", 0.0)))
    fam = _family_name(candidate); fam_onehot = [1.0 if fam == f else 0.0 for f in FAMILIES]
    meta = torch.tensor([
        float(np.clip(sim_score,0,1)),
        float(np.clip(np.log1p(max(length,0.0))/np.log(129.0),0,2)),
        source_score, raw_score, mean_ridge, min_ridge, connects_foreign,
        mean_field, min_field, *fam_onehot
    ], dtype=torch.float32)
    return source_view, path_view, dest_view, meta


def _skeleton_stats(mask: np.ndarray):
    sk = skeletonize(np.asarray(mask, bool)); pts = np.argwhere(sk); endpoints = branches = 0
    if len(pts):
        S = set(map(tuple, pts.tolist()))
        for y, x in S:
            deg = sum((y+dy,x+dx) in S for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx)
            if deg == 1: endpoints += 1
            elif deg >= 3: branches += 1
    return sk, endpoints, branches


def _skeleton_diameter_path(mask: np.ndarray) -> list[tuple[int, int]]:
    sk = skeletonize(np.asarray(mask, bool)); pts = np.argwhere(sk)
    if not len(pts): return []
    S = {tuple(map(int,p)) for p in pts}
    def bfs(start):
        q=deque([start]); parent={start:None}; dist={start:0}; far=start
        while q:
            y,x=q.popleft()
            if dist[(y,x)]>dist[far]: far=(y,x)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==0 and dx==0: continue
                    nb=(y+dy,x+dx)
                    if nb in S and nb not in parent:
                        parent[nb]=(y,x); dist[nb]=dist[(y,x)]+1; q.append(nb)
        return far,parent
    a,_=bfs(next(iter(S))); b,parent=bfs(a); path=[]; cur=b
    while cur is not None: path.append(cur); cur=parent[cur]
    path.reverse(); return path


def build_component_view(record: dict, component_mask: np.ndarray, *, sim_score: float = 0.5, crop_size: int = 33):
    """Build a conservative component-level suppression sample without GT features."""
    image=_as_float_image(record["image"]); prob=np.asarray(record["prob"],np.float32); base=np.asarray(record["base"],bool); comp=np.asarray(component_mask,bool)
    if comp.shape != base.shape or prob.shape != base.shape: raise ValueError("component/base/prob shape mismatch")
    ys,xs=np.where(comp)
    if not len(ys): raise ValueError("empty component")
    cy,cx=int(np.median(ys)),int(np.median(xs)); bh,dark,_=evidence(image); ridge=np.maximum(bh,dark).astype(np.float32); sk,endpoints,branches=_skeleton_stats(comp)
    common=np.dstack([image,np.clip(prob,0,1),base.astype(np.float32),comp.astype(np.float32),np.clip(ridge,0,1),sk.astype(np.float32)]).astype(np.float32)
    view=torch.from_numpy(_crop_center(common,(cy,cx),crop_size).transpose(2,0,1)).float()
    if view.shape[0] != 8: raise AssertionError("unexpected component channel count")
    area=float(comp.sum()); sklen=float(sk.sum()); h=float(ys.max()-ys.min()+1); w=float(xs.max()-xs.min()+1); bbox=max(h*w,1.0); aspect=min(h,w)/max(h,w)
    border=float(bool(comp[0].any() or comp[-1].any() or comp[:,0].any() or comp[:,-1].any())); pvals=prob[comp]; rvals=ridge[comp]
    feat=torch.tensor([
        float(np.clip(np.log1p(area)/np.log1p(comp.size),0,1)),
        float(np.clip(sklen/max(area,1.0),0,1)), float(np.clip(aspect,0,1)), float(np.clip(area/bbox,0,1)),
        float(pvals.mean()) if len(pvals) else 0.0, float(pvals.std()) if len(pvals) else 0.0,
        float(rvals.mean()) if len(rvals) else 0.0, float(rvals.std()) if len(rvals) else 0.0,
        float(np.clip(endpoints/6.0,0,2)), float(np.clip(branches/max(sklen,1.0)*20.0,0,2)), border, float(np.clip(sim_score,0,1))
    ],dtype=torch.float32)
    return view,feat


class DepthwiseBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(); self.dw=nn.Conv2d(cin,cin,3,stride=stride,padding=1,groups=cin,bias=False); self.pw=nn.Conv2d(cin,cout,1,bias=False); self.norm=nn.GroupNorm(4 if cout>=8 else 1,cout); self.act=nn.SiLU(inplace=True)
    def forward(self,x): return self.act(self.norm(self.pw(self.dw(x))))


class CrackEvidenceEncoder(nn.Module):
    def __init__(self,in_channels: int=8,emb_dim: int=48):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(in_channels,16,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,16),nn.SiLU(inplace=True)); self.body=nn.Sequential(DepthwiseBlock(16,24,2),DepthwiseBlock(24,32,2),DepthwiseBlock(32,48,2)); self.proj=nn.Linear(48,emb_dim)
    def forward(self,x):
        z=self.body(self.stem(x)); z=F.adaptive_avg_pool2d(z,1).flatten(1); return F.normalize(self.proj(z),dim=1,eps=1e-6)


class CRCVV55RelationalBlock(nn.Module):
    """64k-parameter same-crack relation block plus conservative component head."""
    def __init__(self,in_channels: int=8,emb_dim: int=48,meta_dim: int=13,component_dim: int=12):
        super().__init__(); self.encoder=CrackEvidenceEncoder(in_channels,emb_dim); rel_dim=emb_dim*7+meta_dim
        self.relation=nn.Sequential(nn.Linear(rel_dim,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.10),nn.Linear(128,64),nn.SiLU())
        self.same_crack=nn.Linear(64,1); self.path_valid=nn.Linear(64,1); self.continuity=nn.Linear(64,1)
        self.component_head=nn.Sequential(nn.Linear(emb_dim+component_dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,1))
    def encode_relation(self,source_view,path_view,destination_view,meta):
        fs=self.encoder(source_view); fp=self.encoder(path_view); fd=self.encoder(destination_view)
        return self.relation(torch.cat([fs,fp,fd,torch.abs(fs-fp),fs*fp,torch.abs(fp-fd),fp*fd,meta],dim=1))
    def forward_recovery(self,source_view,path_view,destination_view,meta):
        z=self.encode_relation(source_view,path_view,destination_view,meta)
        return {"same_crack_logit":self.same_crack(z).squeeze(1),"path_valid_logit":self.path_valid(z).squeeze(1),"continuity_logit":self.continuity(z).squeeze(1)}
    def forward_component(self,component_view,component_features):
        return self.component_head(torch.cat([self.encoder(component_view),component_features],dim=1)).squeeze(1)


def focal_bce_with_logits(logits,targets,*,alpha: float=.75,gamma: float=2.0):
    t=targets.float(); ce=F.binary_cross_entropy_with_logits(logits,t,reduction="none"); p=torch.sigmoid(logits); pt=p*t+(1-p)*(1-t); at=alpha*t+(1-alpha)*(1-t); return (at*(1-pt).pow(gamma)*ce).mean()


def same_source_rank_loss(scores,labels,group_ids,*,margin: float=.30,hard_negatives: int=8):
    scores=scores.flatten(); labels=labels.flatten().bool(); group_ids=group_ids.flatten(); terms=[]
    for g in torch.unique(group_ids):
        m=group_ids==g; pos=scores[m&labels]; neg=scores[m&~labels]
        if not len(pos) or not len(neg): continue
        hard=torch.topk(neg,k=min(int(hard_negatives),len(neg)),largest=True).values
        terms.append(F.softplus(margin-pos[:,None]+hard[None,:]).mean())
    return torch.stack(terms).mean() if terms else scores.sum()*0.0


def recovery_relation_loss(outputs,labels,group_ids,*,rank_weight: float=1.0,same_bce_weight: float=.25,aux_weight: float=.10):
    y=labels.float(); same=outputs["same_crack_logit"]; rank=same_source_rank_loss(same,y,group_ids); bce=focal_bce_with_logits(same,y); pv=focal_bce_with_logits(outputs["path_valid_logit"],y); ct=focal_bce_with_logits(outputs["continuity_logit"],y); total=rank_weight*rank+same_bce_weight*bce+aux_weight*(pv+ct)
    return total,{"rank":float(rank.detach()),"same_bce":float(bce.detach()),"path_valid":float(pv.detach()),"continuity":float(ct.detach())}


def suppression_safety_loss(keep_logits,keep_targets,*,true_crack_weight: float=6.0):
    y=keep_targets.float(); raw=F.binary_cross_entropy_with_logits(keep_logits,y,reduction="none"); w=torch.where(y>.5,torch.full_like(y,true_crack_weight),torch.ones_like(y)); return (raw*w).mean()


@dataclass(frozen=True)
class AbstentionDecision:
    group_id:int; accepted_index:int|None; top_score:float; margin:float


def top1_margin_abstention(scores,group_ids,*,absolute_threshold: float,margin_threshold: float):
    s=np.asarray(scores,np.float64).reshape(-1); g=np.asarray(group_ids).reshape(-1)
    if len(s)!=len(g): raise ValueError("scores and group_ids length mismatch")
    out=[]
    for gid in np.unique(g):
        idx=np.flatnonzero(g==gid); order=idx[np.argsort(-s[idx])]; top=float(s[order[0]]); second=float(s[order[1]]) if len(order)>1 else -np.inf; margin=float(top-second) if np.isfinite(second) else float("inf"); accept=int(order[0]) if top>=absolute_threshold and margin>=margin_threshold else None; out.append(AbstentionDecision(int(gid),accept,top,margin))
    return out
