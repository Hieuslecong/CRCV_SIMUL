from __future__ import annotations
import numpy as np
from .features import build_features
from .policy import add_candidate,TrainingConfig
from .safety import SafetyConfig,project_add,project_remove

def action_scores(heads,image,probability,base_mask):
    X,names=build_features(image,probability,base_mask); z=X.reshape(-1,X.shape[-1]); a=heads["add"].booster_.predict(z).reshape(base_mask.shape).astype(np.float32); r=heads["remove"].booster_.predict(z).reshape(base_mask.shape).astype(np.float32); return a,r,X,names

def refine(image,probability,base_threshold,heads,add_threshold,remove_threshold,safety:SafetyConfig|None=None,training_config:TrainingConfig|None=None):
    """GT-free bidirectional CRCV inference."""
    p=np.asarray(probability,np.float32)
    if not 0<=float(base_threshold)<=1:raise ValueError("bad base threshold")
    b=p>=float(base_threshold); a_score,r_score,X,names=action_scores(heads,image,p,b); cand=add_candidate(p,b,X,names,base_threshold,training_config or TrainingConfig()); rm,rinfo=project_remove(b,r_score,remove_threshold,safety); add,ainfo=project_add(b,cand,a_score,add_threshold,safety); out=(b&~rm)|add
    return out,{"base":b,"add":add,"remove":rm,"add_info":ainfo,"remove_info":rinfo}
