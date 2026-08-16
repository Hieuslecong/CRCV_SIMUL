from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class V53RuntimeConfig:
    recovery_enabled: bool
    verifier_qualified: bool
    suppression_enabled: bool
    width_enabled: bool
    joint_training_enabled: bool
    runtime_policy: str

    @classmethod
    def load(cls, path: str | Path = Path(__file__).resolve().parents[1] / "config_v53.json"):
        d=json.loads(Path(path).read_text())
        return cls(recovery_enabled=bool(d["recovery_enabled"]),verifier_qualified=bool(d["verifier"]["qualified"]),suppression_enabled=bool(d["suppression_enabled"]),width_enabled=bool(d["width_enabled"]),joint_training_enabled=bool(d["joint_training_enabled"]),runtime_policy=str(d["runtime_policy"]))

class CRCVV53Block:
    def __init__(self,cfg:V53RuntimeConfig|None=None):
        self.cfg=cfg or V53RuntimeConfig.load()
        if self.cfg.width_enabled or self.cfg.suppression_enabled or self.cfg.joint_training_enabled:raise RuntimeError("V5.3 release requires suppression/width/joint-training to remain disabled")
        if self.cfg.recovery_enabled and not self.cfg.verifier_qualified:raise RuntimeError("unqualified V5.3 recovery cannot influence runtime output")
    def refine(self, base_mask: np.ndarray):
        base=np.asarray(base_mask).astype(np.uint8)
        if not self.cfg.recovery_enabled:return base.copy(), {"policy":"fail_closed", "recovery_applied":False}
        raise RuntimeError("qualified recovery path is intentionally unavailable in this research release")
