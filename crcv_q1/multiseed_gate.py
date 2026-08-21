from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class MultiSeedSmokeConfig:
    required_seeds: tuple[int, ...] = (1337, 2027, 31415)
    required_backbones: int = 5
    min_active_backbones_per_seed: int = 4
    min_mean_delta_dice_per_seed: float = 0.0
    max_mean_tcrr_per_seed: float = 0.003


def assess_multiseed_smoke(results: Iterable[dict],
                           config: MultiSeedSmokeConfig | None = None) -> dict:
    """Fail-closed development gate for seed robustness at one resolution.

    This is deliberately separate from the publication Q1 gate. It answers only
    whether the current REMOVE subsystem is stable enough across the frozen smoke
    seeds/backbones to justify expanding scientific experiments.
    """
    cfg = config or MultiSeedSmokeConfig()
    by_seed = {}
    for result in results:
        protocol = result.get("protocol", {}) if isinstance(result, dict) else {}
        seed = protocol.get("seed")
        if seed is None:
            continue
        seed = int(seed)
        models = result.get("models", {})
        if not isinstance(models, dict):
            models = {}
        active = 0
        deltas = []
        tcrrs = []
        for model in models.values():
            if model.get("qualification", {}).get("status") == "ACTIVE":
                active += 1
            deployed = model.get("deployed", {})
            dd = deployed.get("delta_dice")
            tr = deployed.get("tcrr")
            if isinstance(dd, (int, float)) and math.isfinite(float(dd)):
                deltas.append(float(dd))
            if isinstance(tr, (int, float)) and math.isfinite(float(tr)):
                tcrrs.append(float(tr))
        by_seed[seed] = {
            "backbones": len(models),
            "active": active,
            "mean_delta_dice": sum(deltas)/len(deltas) if deltas else None,
            "mean_tcrr": sum(tcrrs)/len(tcrrs) if tcrrs else None,
        }

    failures = []
    for seed in cfg.required_seeds:
        state = by_seed.get(seed)
        if state is None:
            failures.append(f"seed {seed} missing")
            continue
        if state["backbones"] < cfg.required_backbones:
            failures.append(f"seed {seed}: backbones {state['backbones']} < {cfg.required_backbones}")
        if state["active"] < cfg.min_active_backbones_per_seed:
            failures.append(f"seed {seed}: active {state['active']} < {cfg.min_active_backbones_per_seed}")
        if (state["mean_delta_dice"] is None or
                state["mean_delta_dice"] < cfg.min_mean_delta_dice_per_seed):
            failures.append(f"seed {seed}: mean delta Dice below floor")
        if (state["mean_tcrr"] is None or
                state["mean_tcrr"] > cfg.max_mean_tcrr_per_seed):
            failures.append(f"seed {seed}: mean TCRR exceeds cap")

    return {
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "per_seed": by_seed,
        "thresholds": {
            "required_seeds": list(cfg.required_seeds),
            "required_backbones": cfg.required_backbones,
            "min_active_backbones_per_seed": cfg.min_active_backbones_per_seed,
            "min_mean_delta_dice_per_seed": cfg.min_mean_delta_dice_per_seed,
            "max_mean_tcrr_per_seed": cfg.max_mean_tcrr_per_seed,
        },
    }
