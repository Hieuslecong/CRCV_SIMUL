from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json

@dataclass(frozen=True)
class Q1Protocol:
    version:str="5.19-q1"
    resolutions:tuple[int,...]=(128,256)
    full_seeds:tuple[int,...]=(1337,2027,31415)
    min_backbones:int=5
    min_datasets:int=3
    min_external_datasets:int=1
    min_cross_dataset_routes:int=2
    require_lobo:bool=True
    require_official_backbone:bool=True
    require_cpu_latency:bool=True
    require_edge_latency:bool=True
    require_final_sealed_until_freeze:bool=True
    primary_metrics:tuple[str,...]=("precision","recall","f1","miou")
    mean_f1_gain_floor:float=.010
    mean_miou_gain_floor:float=.005
    pair_positive_rate_floor:float=.80
    catastrophic_f1_floor:float=-.002
    bootstrap_ci_must_exclude_zero:bool=True
    corrected_p_floor:float=.05

    def validate(self):
        if 256 not in self.resolutions: raise ValueError("256 resolution is mandatory")
        if len(set(self.full_seeds))<3: raise ValueError("at least 3 full seeds required")
        if self.min_backbones<5 or self.min_datasets<3: raise ValueError("Q1 protocol weakened")
        if not 0<self.pair_positive_rate_floor<=1: raise ValueError("bad positive-rate floor")
        return self

def hash_protocol(p:Q1Protocol)->str:
    d=asdict(p); d={k:list(v) if isinstance(v,tuple) else v for k,v in d.items()}
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def save_protocol(path):
    p=Q1Protocol().validate(); d=asdict(p); d["protocol_sha256"]=hash_protocol(p)
    Path(path).write_text(json.dumps(d,indent=2))
    return d
