from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoveQualificationConfig:
    min_delta_dice: float = -0.001
    min_delta_recall: float = -0.002
    max_tcrr: float = 0.003
    min_fprr: float = 0.005


def qualify_remove_policy(metrics: dict,
                          config: RemoveQualificationConfig | None = None) -> dict:
    """Qualification gate evaluated on VAL after CAL threshold selection.

    The gate is intentionally fail-closed. A policy that improves Dice but removes
    too much already-correct crack, or has negligible FP-removal utility, is not
    enabled at runtime. TEST is never used by this function.
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
