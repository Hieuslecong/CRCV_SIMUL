from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoveQualificationConfig:
    min_delta_dice: float = 0.0
    min_delta_recall: float = -0.002
    max_tcrr: float = 0.003
    # FPRR is an effect-size metric, not a safety property. Keeping a hard default
    # FPRR floor made safe low-mutation policies fail on tiny VAL splits.
    min_fprr: float = 0.0
    min_removed_pixels: int = 1


def qualify_remove_policy(metrics: dict,
                          config: RemoveQualificationConfig | None = None) -> dict:
    """Fail-closed REMOVE qualification.

    Safety is determined by non-regressive Dice, bounded Recall loss, and a cap on
    true-crack removal. FPRR remains observable and may optionally have a configured
    floor, but is not treated as a universal safety requirement. If explicit removal
    counts are available, exact no-op behavior is not labelled ACTIVE.
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
        failures.append("false-positive removal rate below configured floor")

    removed = None
    if "removed_pixels" in metrics:
        removed = int(metrics["removed_pixels"])
    elif "true_removed" in metrics and "fp_removed" in metrics:
        removed = int(metrics["true_removed"]) + int(metrics["fp_removed"])
    if removed is not None and removed < cfg.min_removed_pixels:
        failures.append("no effective mutation")

    return {
        "status": "ACTIVE" if not failures else "NO_OP",
        "qualified": not failures,
        "failures": failures,
        "thresholds": {
            "min_delta_dice": cfg.min_delta_dice,
            "min_delta_recall": cfg.min_delta_recall,
            "max_tcrr": cfg.max_tcrr,
            "min_fprr": cfg.min_fprr,
            "min_removed_pixels": cfg.min_removed_pixels,
        },
    }
