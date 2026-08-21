from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Q1Protocol:
    """Publication-evidence protocol for the frozen CRCV V5.18.1 family.

    This protocol intentionally describes evidence requirements, not journal rank.
    A passing gate means the evidence package is complete under this protocol; it
    does not imply acceptance by a Q1 venue.
    """

    version: str = "5.19-q1-v2"
    resolutions: tuple[int, ...] = (128, 256)
    full_seeds: tuple[int, ...] = (1337, 2027, 31415)

    min_backbones: int = 5
    min_reference_backbones: int = 2
    min_datasets: int = 3
    min_external_datasets: int = 1
    min_cross_dataset_routes: int = 2

    require_lobo: bool = True
    require_published_refiner: bool = True
    require_artifact_provenance: bool = True
    require_historical_exposure_guard: bool = True
    require_cluster_statistics: bool = True
    require_runtime_gt_mutation_test: bool = True
    require_cpu_latency: bool = True
    require_edge_latency_if_claimed: bool = True
    require_final_sealed_until_freeze: bool = True

    primary_metrics: tuple[str, ...] = ("dice", "crack_iou")
    mean_dice_gain_floor: float = 0.010
    mean_crack_iou_gain_floor: float = 0.005
    pair_positive_rate_floor: float = 0.80
    catastrophic_dice_floor: float = -0.002
    bootstrap_ci_must_exclude_zero: bool = True
    corrected_p_floor: float = 0.05

    def validate(self) -> "Q1Protocol":
        if 256 not in self.resolutions:
            raise ValueError("native 256 resolution is mandatory")
        if tuple(dict.fromkeys(self.full_seeds)) != self.full_seeds or len(self.full_seeds) < 3:
            raise ValueError("full_seeds must contain at least three unique frozen seeds")
        if self.min_backbones < 5:
            raise ValueError("at least five backbones are required")
        if self.min_reference_backbones < 2:
            raise ValueError("at least two canonical/reference backbones are required")
        if self.min_datasets < 3 or self.min_external_datasets < 1:
            raise ValueError("multi-dataset and external evidence cannot be weakened")
        if self.min_cross_dataset_routes < 2:
            raise ValueError("at least two cross-dataset routes are required")
        if not 0 < self.pair_positive_rate_floor <= 1:
            raise ValueError("bad positive-pair floor")
        if not 0 < self.corrected_p_floor < 1:
            raise ValueError("bad corrected-p threshold")
        return self


def _jsonable_protocol(p: Q1Protocol) -> dict:
    d = asdict(p)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


def hash_protocol(p: Q1Protocol) -> str:
    payload = json.dumps(
        _jsonable_protocol(p), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def save_protocol(path: str | Path) -> dict:
    p = Q1Protocol().validate()
    d = _jsonable_protocol(p)
    d["protocol_sha256"] = hash_protocol(p)
    Path(path).write_text(json.dumps(d, indent=2) + "\n")
    return d
