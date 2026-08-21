from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoveQualificationConfig:
    min_delta_dice: float = 0.0
    min_delta_recall: float = -0.002
    max_tcrr: float = 0.003
    min_fprr: float = 0.005


def qualify_remove_policy(metrics: dict,
                          config: RemoveQualificationConfig | None = None) -> dict:
    """Fail-closed VAL qualification after CAL threshold selection.

    A policy is enabled only when it is non-regressive in Dice, keeps Recall loss
    within the safety floor, removes at most a small fraction of already-correct
    crack (TCRR), and removes enough false-positive mass to justify mutation.
    TEST must never be used by this gate.
    """
    cfg = config or RemoveQualificationConfig()
    required = ("delta_dice", "delta_recall", "tcrr", "fprr")
    missing = [k for k in required if k not in metrics]
    if missing:
        return {"status":"NO_OP", "qualified":False, "failures":[f"missing {k}" for k in missing]}
    failures = []
    if float(metrics["delta_dice"]) < cfg.min_delta_dice:
        failures.append("delta_dice below floor")
    if float(metrics["delta_recall"]) < cfg.min_delta_recall:
        failures.append("recall loss exceeds safety floor")
    if float(metrics["tcrr"]) > cfg.max_tcrr:
        failures.append("true-crack removal rate exceeds safety cap")
    if float(metrics["fprr"]) < cfg.min_fprr:
        failures.append("false-positive removal rate too small to justify mutation")
    return {
        "status": "ACTIVE" if not failures else "NO_OP",
        "qualified": not failures,
        "failures": failures,
        "thresholds": {
            "min_delta_dice": cfg.min_delta_dice,
            "min_delta_recall": cfg.min_delta_recall,
            "max_tcrr": cfg.max_tcrr,
            "min_fprr": cfg.min_fprr,
        },
    }
