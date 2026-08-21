from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

from crcv_q1.remove_qualification import RemoveQualificationConfig, qualify_remove_policy


@dataclass(frozen=True)
class OperatingPoint:
    family: str
    parameters: Any
    metrics: dict
    # Retained for backward-compatible reporting only. Selection no longer uses an
    # arbitrary weighted sum of Dice/IoU/FPRR/TCRR.
    utility: float = 0.0


def _safety_effect_key(metrics: dict, conservative_tiebreak: float = 0.0):
    """Lexicographic constrained-selection key (smaller is better).

    All inputs reaching this key have already passed the qualification constraints.
    We then minimize true-crack damage, maximize useful FP removal, maximize Dice
    and Crack-IoU gains, and finally prefer the caller-provided conservative tie
    break. This avoids scientifically unjustified weighted utility coefficients.
    """
    return (
        float(metrics["tcrr"]),
        -float(metrics["fprr"]),
        -float(metrics["delta_dice"]),
        -float(metrics.get("delta_crack_iou", 0.0)),
        -float(conservative_tiebreak),
    )


def correction_utility(metrics: dict) -> float:
    """Deprecated scalar retained for report compatibility.

    It is intentionally equal to Dice gain only and is NOT used for operating-point
    selection. New code should inspect the metrics and qualification constraints.
    """
    return float(metrics["delta_dice"])


def select_conservative_pixel_threshold(evaluate: Callable[[float], dict],
                                        thresholds: Iterable[float],
                                        qualification: RemoveQualificationConfig | None = None):
    """Select a CAL-qualified pixel threshold by constrained lexicographic order."""
    qcfg = qualification or RemoveQualificationConfig()
    valid = []
    for threshold in thresholds:
        threshold = float(threshold)
        metrics = evaluate(threshold)
        if qualify_remove_policy(metrics, qcfg)["qualified"]:
            # Higher threshold is the final tie-break because it mutates less.
            valid.append((_safety_effect_key(metrics, threshold), threshold, metrics))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    _, threshold, metrics = valid[0]
    return OperatingPoint("pixel", threshold, metrics, correction_utility(metrics))


def select_component_configuration(evaluate: Callable[[Any], dict],
                                   configurations: Iterable[Any],
                                   qualification: RemoveQualificationConfig | None = None):
    """Select a CAL-qualified scale-free component configuration."""
    qcfg = qualification or RemoveQualificationConfig()
    valid = []
    for config in configurations:
        metrics = evaluate(config)
        if qualify_remove_policy(metrics, qcfg)["qualified"]:
            valid.append((_safety_effect_key(metrics), config, metrics))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    _, config, metrics = valid[0]
    return OperatingPoint("component", config, metrics, correction_utility(metrics))


def select_policy_family(*candidates: OperatingPoint | None):
    """CAL-only family selection under the same constrained order."""
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: _safety_effect_key(c.metrics))
