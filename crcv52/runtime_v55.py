from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class V55RuntimeConfig:
    proposal_qualified: bool
    relation_verifier_qualified: bool
    suppression_qualified: bool
    recovery_enabled: bool
    suppression_enabled: bool
    width_enabled: bool
    joint_training_enabled: bool
    runtime_policy: str
    max_add_fraction: float
    max_remove_fraction: float
    max_cc_increase: int

    @classmethod
    def load(cls, path: str | Path = Path(__file__).resolve().parents[1] / "config_v55.json") -> "V55RuntimeConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8")); safety = d["runtime_safety"]
        return cls(
            proposal_qualified=bool(d["proposal"]["qualified"]),
            relation_verifier_qualified=bool(d["relation_verifier"]["qualified"]),
            suppression_qualified=bool(d["suppression_head"]["qualified"]),
            recovery_enabled=bool(d["recovery_enabled"]),
            suppression_enabled=bool(d["suppression_enabled"]),
            width_enabled=bool(d["width_enabled"]),
            joint_training_enabled=bool(d["joint_training_enabled"]),
            runtime_policy=str(d["runtime_policy"]),
            max_add_fraction=float(safety["max_add_fraction"]),
            max_remove_fraction=float(safety["max_remove_fraction"]),
            max_cc_increase=int(safety["max_cc_increase"]),
        )


def _cc_count(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(np.asarray(mask, np.uint8), 8)
    return max(int(n)-1, 0)


class StructuralSafetyGate:
    """Last fail-safe after learned decisions; not a performance selector."""
    def __init__(self, *, max_add_fraction: float=.02, max_remove_fraction: float=.01, max_cc_increase: int=0):
        self.max_add_fraction=float(max_add_fraction); self.max_remove_fraction=float(max_remove_fraction); self.max_cc_increase=int(max_cc_increase)

    def validate(self, base: np.ndarray, *, add: np.ndarray|None=None, remove: np.ndarray|None=None):
        b=np.asarray(base,bool); a=np.zeros_like(b) if add is None else np.asarray(add,bool); r=np.zeros_like(b) if remove is None else np.asarray(remove,bool)
        if a.shape!=b.shape or r.shape!=b.shape: raise ValueError("add/remove shape mismatch")
        a &= ~b; r &= b
        base_pixels=max(int(b.sum()),1); add_fraction=float(a.sum()/base_pixels); remove_fraction=float(r.sum()/base_pixels); reasons=[]
        if add_fraction>self.max_add_fraction: reasons.append("add_fraction")
        if remove_fraction>self.max_remove_fraction: reasons.append("remove_fraction")
        if a.any():
            n_add,lab=cv2.connectedComponents(a.astype(np.uint8),8)
            for cid in range(1,n_add):
                comp=lab==cid
                if not np.any(cv2.dilate(comp.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool)&b):
                    reasons.append("isolated_add"); break
        refined=(b&~r)|a; base_cc=_cc_count(b); refined_cc=_cc_count(refined)
        if refined_cc>base_cc+self.max_cc_increase: reasons.append("cc_increase")
        ok=not reasons
        if not ok: refined=b.copy(); a=np.zeros_like(b); r=np.zeros_like(b)
        return refined.astype(np.uint8),{"safety_pass":bool(ok),"reasons":reasons,"add_pixels":int(a.sum()),"remove_pixels":int(r.sum()),"add_fraction":add_fraction,"remove_fraction":remove_fraction,"base_cc":base_cc,"refined_cc":_cc_count(refined)}


class CRCVV55Block:
    """Simulation-informed relation block with exact fail-closed behavior."""
    def __init__(self,cfg:V55RuntimeConfig|None=None,*,recovery_fn:Callable[[dict],np.ndarray]|None=None,suppression_fn:Callable[[dict],np.ndarray]|None=None):
        self.cfg=cfg or V55RuntimeConfig.load(); self.recovery_fn=recovery_fn; self.suppression_fn=suppression_fn
        if self.cfg.width_enabled or self.cfg.joint_training_enabled: raise RuntimeError("V5.5 keeps width and Base joint-training disabled")
        if self.cfg.recovery_enabled and (not self.cfg.proposal_qualified or not self.cfg.relation_verifier_qualified): raise RuntimeError("unqualified recovery cannot affect runtime output")
        if self.cfg.suppression_enabled and not self.cfg.suppression_qualified: raise RuntimeError("unqualified suppression cannot affect runtime output")
        if self.cfg.recovery_enabled and self.recovery_fn is None: raise RuntimeError("qualified recovery requires a recovery_fn")
        if self.cfg.suppression_enabled and self.suppression_fn is None: raise RuntimeError("qualified suppression requires a suppression_fn")
        self.safety=StructuralSafetyGate(max_add_fraction=self.cfg.max_add_fraction,max_remove_fraction=self.cfg.max_remove_fraction,max_cc_increase=self.cfg.max_cc_increase)

    def refine(self,base_mask:np.ndarray,*,record:dict|None=None):
        base=np.asarray(base_mask).astype(bool)
        if not self.cfg.recovery_enabled and not self.cfg.suppression_enabled:
            return base.astype(np.uint8),{"policy":"fail_closed","proposal_qualified":self.cfg.proposal_qualified,"relation_verifier_qualified":self.cfg.relation_verifier_qualified,"suppression_qualified":self.cfg.suppression_qualified,"recovery_applied":False,"suppression_applied":False}
        if record is None: raise ValueError("record is required when a V5.5 correction head is enabled")
        add=self.recovery_fn(record) if self.cfg.recovery_enabled else None; remove=self.suppression_fn(record) if self.cfg.suppression_enabled else None
        refined,safety=self.safety.validate(base,add=add,remove=remove)
        return refined,{"policy":self.cfg.runtime_policy,"recovery_applied":bool(self.cfg.recovery_enabled and safety["safety_pass"]),"suppression_applied":bool(self.cfg.suppression_enabled and safety["safety_pass"]),**safety}
