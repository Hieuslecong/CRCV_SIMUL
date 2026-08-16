from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class V54RuntimeConfig:
    recovery_enabled: bool
    proposal_qualified: bool
    verifier_qualified: bool
    suppression_enabled: bool
    width_enabled: bool
    joint_training_enabled: bool
    runtime_policy: str
    @classmethod
    def load(cls,path: str|Path=Path(__file__).resolve().parents[1]/'config_v54.json'):
        d=json.loads(Path(path).read_text());return cls(bool(d['recovery_enabled']),bool(d['proposal_qualification']['qualified']),bool(d['verifier']['qualified']),bool(d['suppression_enabled']),bool(d['width_enabled']),bool(d['joint_training_enabled']),str(d['runtime_policy']))

class CRCVV54Block:
    """V5.4 fail-closed runtime. Proposal is qualified; final acceptance is not."""
    def __init__(self,cfg:V54RuntimeConfig|None=None):
        self.cfg=cfg or V54RuntimeConfig.load()
        if self.cfg.suppression_enabled or self.cfg.width_enabled or self.cfg.joint_training_enabled:
            raise RuntimeError('V5.4 release requires suppression/width/joint-training disabled')
        if self.cfg.recovery_enabled and (not self.cfg.proposal_qualified or not self.cfg.verifier_qualified):
            raise RuntimeError('unqualified V5.4 recovery cannot affect runtime output')
    def refine(self,base_mask:np.ndarray):
        base=np.asarray(base_mask).astype(np.uint8)
        if not self.cfg.recovery_enabled:
            return base.copy(),{'policy':'fail_closed','proposal_qualified':self.cfg.proposal_qualified,'verifier_qualified':self.cfg.verifier_qualified,'recovery_applied':False}
        raise RuntimeError('qualified recovery path not available in this research release')
