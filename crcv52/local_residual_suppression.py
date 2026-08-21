from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class LocalResidualSuppressionConfig:
    authenticity_threshold: float = 0.03
    max_region_fraction: float = 0.025
    max_total_remove_fraction: float = 0.03
    max_foreground_remove_fraction: float = 0.20
    min_region_pixels: int = 2
    connectivity: int = 8

    def validate(self) -> None:
        if not 0.0 <= self.authenticity_threshold <= 1.0:
            raise ValueError("authenticity_threshold must be in [0, 1]")
        if not 0.0 < self.max_region_fraction <= 1.0:
            raise ValueError("max_region_fraction must be in (0, 1]")
        if not 0.0 < self.max_total_remove_fraction <= 1.0:
            raise ValueError("max_total_remove_fraction must be in (0, 1]")
        if not 0.0 < self.max_foreground_remove_fraction <= 1.0:
            raise ValueError("max_foreground_remove_fraction must be in (0, 1]")
        if self.min_region_pixels < 1:
            raise ValueError("min_region_pixels must be >= 1")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


def local_residual_suppress(
    base_mask: np.ndarray,
    authenticity_map: np.ndarray,
    config: LocalResidualSuppressionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove very-low-authenticity residuals inside Frozen Base support only.

    The deletion budget is bounded both by image area and predicted-foreground
    area. This avoids an image-relative cap becoming permissive when the crack
    foreground is extremely sparse. Runtime intentionally accepts no GT input.
    """
    cfg = config or LocalResidualSuppressionConfig()
    cfg.validate()

    base = np.asarray(base_mask, dtype=bool)
    auth = np.asarray(authenticity_map, dtype=np.float32)
    if base.ndim != 2 or auth.ndim != 2:
        raise ValueError("base_mask and authenticity_map must be 2-D")
    if base.shape != auth.shape:
        raise ValueError("base_mask/authenticity_map shape mismatch")

    if not base.any():
        return base.copy(), np.zeros_like(base, dtype=bool)

    risk = base & np.isfinite(auth) & (auth < float(cfg.authenticity_threshold))
    n_labels, labels = cv2.connectedComponents(
        risk.astype(np.uint8), connectivity=cfg.connectivity
    )

    refined = base.copy()
    removed = np.zeros_like(base, dtype=bool)
    base_pixels = int(base.sum())
    max_region = max(1, int(np.floor(cfg.max_region_fraction * base.size)))
    image_budget = max(1, int(np.floor(cfg.max_total_remove_fraction * base.size)))
    foreground_budget = max(
        1, int(np.floor(cfg.max_foreground_remove_fraction * base_pixels))
    )
    max_total = min(image_budget, foreground_budget)
    total_removed = 0

    for component_id in range(1, n_labels):
        region = labels == component_id
        area = int(region.sum())
        if area < cfg.min_region_pixels or area > max_region:
            continue
        if total_removed + area > max_total:
            continue
        region &= refined
        area = int(region.sum())
        if area < cfg.min_region_pixels:
            continue
        refined[region] = False
        removed[region] = True
        total_removed += area

    if np.any(removed & ~base):
        raise AssertionError("removed_mask escaped Frozen Base support")
    if np.any(refined & ~base):
        raise AssertionError("suppression created foreground outside Frozen Base")
    if int(removed.sum()) > max_total:
        raise AssertionError("suppression exceeded the conservative deletion budget")
    return refined, removed


def suppression_statistics(base_mask: np.ndarray, refined_mask: np.ndarray) -> dict:
    base = np.asarray(base_mask, dtype=bool)
    refined = np.asarray(refined_mask, dtype=bool)
    if base.shape != refined.shape:
        raise ValueError("shape mismatch")
    removed = base & ~refined
    base_pixels = int(base.sum())
    removed_pixels = int(removed.sum())
    return {
        "base_pixels": base_pixels,
        "removed_pixels": removed_pixels,
        "removed_fraction_of_base": float(removed_pixels / base_pixels) if base_pixels else 0.0,
    }
