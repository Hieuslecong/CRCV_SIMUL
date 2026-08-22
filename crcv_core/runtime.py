from __future__ import annotations
import numpy as np
from .features import build_features
from .policy import add_candidate,TrainingConfig
from .safety import SafetyConfig,project_add,project_remove

def _predict(head,z):
    model=head.booster_ if hasattr(head,"booster_") else head
    if not hasattr(model,"predict"):raise TypeError("action head must provide predict")
    return np.asarray(model.predict(z),np.float32)

def action_scores(heads,image,probability,base_mask):
    if not isinstance(heads,dict) or not {"add","remove"}<=set(heads):raise ValueError("ADD/REMOVE heads required")
    X,names=build_features(image,probability,base_mask); z=X.reshape(-1,X.shape[-1]); a=_predict(heads["add"],z).reshape(base_mask.shape); r=_predict(heads["remove"],z).reshape(base_mask.shape); return a,r,X,names

def refine(image,probability,base_threshold,heads,add_threshold,remove_threshold,safety:SafetyConfig|None=None,training_config:TrainingConfig|None=None,qualified:bool=False):
    """GT-free inference; unqualified policies fail closed to exact Base."""
    p=np.asarray(probability,np.float32)
    if not np.isfinite(base_threshold) or not 0<=float(base_threshold)<=1:raise ValueError("bad base threshold")
    if p.ndim!=2 or not np.isfinite(p).all() or (p.size and (float(p.min())<0 or float(p.max())>1)):raise ValueError("bad probability")
    b=p>=float(base_threshold)
    if qualified is not True:
        z=np.zeros_like(b); return b.copy(),{"base":b,"add":z,"remove":z,"status":"NO_OP_UNQUALIFIED"}
    a_score,r_score,X,names=action_scores(heads,image,p,b); cand=add_candidate(p,b,X,names,base_threshold,training_config or TrainingConfig()); rm,rinfo=project_remove(b,r_score,remove_threshold,safety); add,ainfo=project_add(b,cand,a_score,add_threshold,safety); out=(b&~rm)|add
    return out,{"base":b,"add":add,"remove":rm,"add_info":ainfo,"remove_info":rinfo,"status":"ACTIVE"}
