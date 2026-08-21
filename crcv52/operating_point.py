from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

from crcv_q1.remove_qualification import RemoveQualificationConfig, qualify_remove_policy


@dataclass(frozen=True)
class OperatingPoint:
    family: str
    parameters: Any
    metrics: dict
    utility: float


def correction_utility(metrics: dict) -> float:
    return (float(metrics["delta_dice"])
            + 0.25 * float(metrics.get("delta_crack_iou", 0.0))
            + 0.02 * float(metrics["fprr"])
            - 0.50 * float(metrics["tcrr"]))


def select_conservative_pixel_threshold(evaluate: Callable[[float], dict],
                                        thresholds: Iterable[float],
                                        qualification: RemoveQualificationConfig | None = None):
    """Select a CAL-safe pixel threshold with a safety-first tie-break.

    Among CAL-qualified thresholds, prefer the highest threshold (least mutation),
    then utility. This avoids choosing an aggressive operating point merely because
    it gives slightly better CAL Dice.
    """
    qcfg = qualification or RemoveQualificationConfig()
    valid = []
    for threshold in thresholds:
        metrics = evaluate(float(threshold))
        if qualify_remove_policy(metrics, qcfg)["qualified"]:
            valid.append((float(threshold), correction_utility(metrics), metrics))
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1]), reverse=True)
    threshold, utility, metrics = valid[0]
    return OperatingPoint("pixel", threshold, metrics, utility)


def select_component_configuration(evaluate: Callable[[Any], dict],
                                   configurations: Iterable[Any],
                                   qualification: RemoveQualificationConfig | None = None):
    """Select the highest-utility CAL-qualified whole-component configuration."""
    qcfg = qualification or RemoveQualificationConfig()
    valid = []
    for config in configurations:
        metrics = evaluate(config)
        if qualify_remove_policy(metrics, qcfg)["qualified"]:
            valid.append((correction_utility(metrics), config, metrics))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    utility, config, metrics = valid[0]
    return OperatingPoint("component", config, metrics, utility)


def select_policy_family(*candidates: OperatingPoint | None):
    """CAL-only family selection. VAL is reserved for fail-closed qualification."""
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    return max(valid, key=lambda c: c.utility)
