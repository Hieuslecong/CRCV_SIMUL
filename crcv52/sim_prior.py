from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import parse_xy


_EPS = 1e-8


def resample_polyline(points: np.ndarray, *, min_points: int = 6, max_points: int = 256) -> np.ndarray:
    """Arc-length resampling used before estimating simulation morphology.

    Simulation traces often contain repeated or nearly repeated points. Scoring raw
    turning angles would therefore be dominated by meshing/sampling artefacts rather
    than crack morphology. Each trajectory is resampled at its own median non-zero
    step length and capped to a bounded number of points.
    """
    p = np.asarray(points, np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if len(p) < 2:
        return p.copy()

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.r_[True, seg > _EPS]
    p = p[keep]
    if len(p) < 2:
        return p.copy()

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    nz = seg[seg > _EPS]
    if not len(nz):
        return p[:1].copy()
    step = float(np.median(nz))
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    if total <= _EPS:
        return p[:1].copy()

    n = int(np.clip(int(total / max(step, _EPS)) + 1, min_points, max_points))
    q = np.linspace(0.0, total, n, dtype=np.float32)
    out = np.c_[
        np.interp(q, cum, p[:, 0]),
        np.interp(q, cum, p[:, 1]),
    ].astype(np.float32)
    return out


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = max(1.4826 * mad, 1e-3)
    return med, scale


def _turn_angles(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float32)
    if len(p) < 3:
        return np.empty(0, np.float32)
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    good = n > _EPS
    d = d[good]
    n = n[good]
    if len(d) < 2:
        return np.empty(0, np.float32)
    u = d / (n[:, None] + _EPS)
    dot = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return np.arccos(dot).astype(np.float32)


def polyline_tortuosity(points: np.ndarray) -> float:
    p = np.asarray(points, np.float32)
    if len(p) < 2:
        return 1.0
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(p[-1] - p[0]))
    return float(path / max(chord, _EPS))


@dataclass(frozen=True)
class SimulationPriorProfile:
    schema: str
    trajectories: int
    resampled_trajectories: int
    turn_samples: int
    log_turn_median: float
    log_turn_scale: float
    log_tortuosity_median: float
    log_tortuosity_scale: float
    turn_q01: float
    turn_q05: float
    turn_q50: float
    turn_q95: float
    turn_q99: float
    tort_q01: float
    tort_q50: float
    tort_q95: float
    tort_q99: float

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationPriorProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_simulation_prior(sequences: Iterable[np.ndarray]) -> SimulationPriorProfile:
    turns: list[float] = []
    torts: list[float] = []
    total = 0
    used = 0
    for seq in sequences:
        total += 1
        r = resample_polyline(np.asarray(seq, np.float32))
        if len(r) < 6:
            continue
        used += 1
        a = _turn_angles(r)
        if len(a):
            turns.extend(a.tolist())
        torts.append(polyline_tortuosity(r))

    if not turns or not torts:
        raise ValueError("simulation trajectories do not contain enough valid geometry")

    ta = np.asarray(turns, np.float64)
    tr = np.asarray(torts, np.float64)
    tm, ts = _robust_location_scale(np.log1p(ta))
    rm, rs = _robust_location_scale(np.log1p(np.maximum(tr - 1.0, 0.0)))
    tq = np.quantile(ta, [0.01, 0.05, 0.50, 0.95, 0.99])
    rq = np.quantile(tr, [0.01, 0.50, 0.95, 0.99])
    return SimulationPriorProfile(
        schema="crcv-v5.5-simulation-geometry-prior-1",
        trajectories=int(total),
        resampled_trajectories=int(used),
        turn_samples=int(len(ta)),
        log_turn_median=float(tm),
        log_turn_scale=float(ts),
        log_tortuosity_median=float(rm),
        log_tortuosity_scale=float(rs),
        turn_q01=float(tq[0]),
        turn_q05=float(tq[1]),
        turn_q50=float(tq[2]),
        turn_q95=float(tq[3]),
        turn_q99=float(tq[4]),
        tort_q01=float(rq[0]),
        tort_q50=float(rq[1]),
        tort_q95=float(rq[2]),
        tort_q99=float(rq[3]),
    )


def fit_simulation_prior_from_xy(path: str | Path) -> SimulationPriorProfile:
    """Fit a geometry-only prior from XY traces separated by literal ``0,0``."""
    return fit_simulation_prior(parse_xy(path))


class SimulationGeometryPrior:
    """A bounded crack-morphology plausibility prior.

    The score is intentionally weak: it may rank candidate geometry, but it must never
    by itself authorize suppression or recovery. Real RGB and natural OOF evidence are
    required by V5.5's downstream relation heads.
    """

    def __init__(self, profile: SimulationPriorProfile):
        self.profile = profile

    @classmethod
    def load(cls, path: str | Path) -> "SimulationGeometryPrior":
        return cls(SimulationPriorProfile.from_json(path))

    @staticmethod
    def _robust_score(z: np.ndarray) -> np.ndarray:
        z = np.abs(np.asarray(z, np.float32))
        penalty = np.where(z <= 1.0, 0.15 * z * z, 0.15 + 0.45 * (z - 1.0))
        return np.exp(-np.clip(penalty, 0.0, 8.0)).astype(np.float32)

    def score_polyline(self, points: np.ndarray) -> float:
        r = resample_polyline(points)
        if len(r) < 6:
            return 0.5
        a = _turn_angles(r)
        if not len(a):
            turn_score = 1.0
        else:
            z = (np.log1p(a) - self.profile.log_turn_median) / max(
                self.profile.log_turn_scale, 1e-3
            )
            s = np.clip(self._robust_score(z), 1e-6, 1.0)
            turn_score = float(np.exp(np.mean(np.log(s))))
        tort = polyline_tortuosity(r)
        rz = (
            np.log1p(max(tort - 1.0, 0.0)) - self.profile.log_tortuosity_median
        ) / max(self.profile.log_tortuosity_scale, 1e-3)
        tort_score = float(self._robust_score(np.asarray([rz]))[0])
        return float(np.clip(0.75 * turn_score + 0.25 * tort_score, 0.0, 1.0))

    def candidate_score(self, candidate: dict) -> float:
        path_yx = candidate.get("path_yx")
        if path_yx is None or len(path_yx) < 2:
            return 0.5
        xy = np.asarray([(float(x), float(y)) for y, x in path_yx], np.float32)
        return self.score_polyline(xy)
