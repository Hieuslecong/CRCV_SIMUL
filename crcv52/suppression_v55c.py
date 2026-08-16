from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentScore:
    component_id: int
    area: int
    mean_probability: float


def enumerate_runtime_components(record: dict) -> list[ComponentScore]:
    """Enumerate *all* Base-positive connected components without GT access.

    This function is the only legal runtime component enumeration for V5.5c.
    It deliberately ignores ``record['gt']`` even when a caller supplies it.
    """
    base = np.asarray(record["base"], bool)
    prob = np.asarray(record["prob"], np.float32)
    if base.shape != prob.shape:
        raise ValueError("base/prob shape mismatch")
    n, lab = cv2.connectedComponents(base.astype(np.uint8), 8)
    out: list[ComponentScore] = []
    for cid in range(1, n):
        comp = lab == cid
        vals = prob[comp]
        out.append(
            ComponentScore(
                component_id=int(cid),
                area=int(comp.sum()),
                mean_probability=float(vals.mean()) if len(vals) else 0.0,
            )
        )
    return out


def confidence_suppression_mask(record: dict, *, mean_probability_threshold: float) -> np.ndarray:
    """Return a remove mask using only Base probability and Base connectivity.

    A complete connected component is removable when its mean Base probability is
    less than or equal to the CAL-selected threshold. No GT, simulation label, or
    recovery-verifier state is consulted.
    """
    base = np.asarray(record["base"], bool)
    _, lab = cv2.connectedComponents(base.astype(np.uint8), 8)
    remove = np.zeros_like(base, bool)
    for z in enumerate_runtime_components(record):
        if z.mean_probability <= float(mean_probability_threshold):
            remove |= lab == z.component_id
    return remove


def component_pixel_metrics(records: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    """CAL-only helper: all runtime components plus TP/FP pixel counts.

    GT is used only here to measure calibration safety. Component eligibility is
    still produced by ``enumerate_runtime_components`` so CAL and runtime candidate
    spaces are identical.
    """
    scores, metrics = [], []
    for record in records:
        base = np.asarray(record["base"], bool)
        gt = np.asarray(record["gt"], bool)
        _, lab = cv2.connectedComponents(base.astype(np.uint8), 8)
        for z in enumerate_runtime_components(record):
            comp = lab == z.component_id
            scores.append(z.mean_probability)
            metrics.append((int((comp & gt).sum()), int((comp & ~gt).sum())))
    return np.asarray(scores, np.float64), np.asarray(metrics, np.float64)


def calibrate_confidence_suppression(
    records: Iterable[dict],
    *,
    max_true_pixel_removal: float = 0.01,
    target_false_pixel_removal: float = 0.30,
    grid_points: int = 250,
) -> tuple[dict, bool]:
    scores, metrics = component_pixel_metrics(records)
    if not len(scores):
        raise RuntimeError("empty CAL component set")
    total_true = float(metrics[:, 0].sum())
    total_false = float(metrics[:, 1].sum())
    grid = np.unique(np.quantile(scores, np.linspace(0.001, 0.95, int(grid_points))))
    rows = []
    for t in grid:
        rem = scores <= t
        tr = float(metrics[rem, 0].sum()) / (total_true + 1e-9)
        fr = float(metrics[rem, 1].sum()) / (total_false + 1e-9)
        rows.append(
            {
                "mean_probability_threshold": float(t),
                "true_pixel_removal": tr,
                "false_pixel_removal": fr,
                "removed_components": int(rem.sum()),
            }
        )
    safe = [r for r in rows if r["true_pixel_removal"] <= max_true_pixel_removal]
    if not safe:
        best = min(rows, key=lambda r: r["true_pixel_removal"])
        return best, False
    best = max(safe, key=lambda r: (r["false_pixel_removal"], -r["true_pixel_removal"]))
    return best, bool(best["false_pixel_removal"] >= target_false_pixel_removal)
