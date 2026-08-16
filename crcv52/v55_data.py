from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .relational_v55 import (
    _ordered_add_path,
    _skeleton_diameter_path,
    build_component_view,
    build_relation_views,
)
from .sim_prior import SimulationGeometryPrior
from .gap_metrics import core_gap_hit, core_missing_skeleton


def unpack_add(record: dict) -> np.ndarray:
    bits = np.frombuffer(record["add_pack"], dtype=np.uint8)
    n = int(np.prod(record["shape"]))
    return np.unpackbits(bits)[:n].reshape(record["shape"]).astype(bool)


def source_from_key(source_key) -> tuple[int, int]:
    """Recover the source endpoint from V5.4's compact source key."""
    k = source_key
    if isinstance(k, list):
        k = tuple(k)
    if isinstance(k, tuple) and len(k) >= 2:
        if k[0] == "s":
            y, x = k[1]
            return int(y), int(x)
        if k[0] == "p":
            # Pair keys are rejected from the active V5.4 proposal set, but remain
            # parseable for audit/reproduction.
            y, x = k[1]
            return int(y), int(x)
    raise ValueError(f"unsupported source_key: {source_key!r}")


def compact_row_to_candidate(row: dict) -> dict:
    add = unpack_add(row)
    source = source_from_key(row["source_key"])
    path = _ordered_add_path(add, source)
    return {
        "family": row.get("family", "other"),
        "add": add,
        "path_yx": path,
        "source_yx": source,
        "source_endpoint_yx": source,
        "length": float(row.get("length", int(add.sum()))),
        "score": float(row.get("raw_score", 0.0)),
        "source_score": float(row.get("source_score", 0.0)),
        "mean_ridge": float(row.get("evidence", 0.0)),
        "mean_field": float(row.get("evidence", 0.0)),
        "connects_foreign": bool("connect" in str(row.get("family", "")).lower()),
    }


def load_v54_coregap_bank(
    artifacts: str | Path,
    split: str,
) -> tuple[dict[str, dict], list[dict]]:
    """Load the frozen V5.4 top-k bank and corresponding frozen Base records."""
    if split not in {"module", "cal"}:
        raise ValueError("split must be module or cal")
    A = Path(artifacts)
    base_file = A / "cache" / ("v52_module.pkl" if split == "module" else "v52_cal.pkl")
    part_dir = A / "cache" / f"v54_{split}_coregap_topk_parts"
    if not base_file.exists():
        raise FileNotFoundError(base_file)
    if not part_dir.exists():
        raise FileNotFoundError(part_dir)
    base = {r["name"]: r for r in pickle.load(open(base_file, "rb"))}
    rows: list[dict] = []
    for p in sorted(part_dir.glob("*.pkl")):
        rows.extend(pickle.load(open(p, "rb"))["records"])
    return base, rows


class RelationBankDataset(Dataset):
    """On-the-fly crops from the frozen V5.4 candidate bank.

    GT is used only for the persisted V5.4 labels/metrics. It is never included in the
    source/path/destination image tensors.
    """

    def __init__(
        self,
        base_records: dict[str, dict],
        rows: Iterable[dict],
        sim_prior: SimulationGeometryPrior,
        *,
        crop_size: int = 33,
        include_ambiguous: bool = False,
    ):
        self.base = base_records
        self.rows = [
            r for r in rows if include_ambiguous or int(r.get("label", -1)) >= 0
        ]
        self.prior = sim_prior
        self.crop_size = int(crop_size)
        self.group_map: dict[tuple, int] = {}
        self._cache: dict[int, tuple] = {}
        for r in self.rows:
            g = (r["image"], repr(r["source_key"]))
            if g not in self.group_map:
                self.group_map[g] = len(self.group_map)
        self.image_names = [str(r["image"]) for r in self.rows]
        self.group_ids = np.asarray(
            [self.group_map[(r["image"], repr(r["source_key"]))] for r in self.rows],
            dtype=np.int64,
        )
        self.core_total_by_image: dict[str, int] = {}
        for name, rec in self.base.items():
            if str(rec.get("typ", "")) == "normal":
                self.core_total_by_image[name] = 0
            else:
                ms = core_missing_skeleton(
                    np.asarray(rec["gt"], bool),
                    np.asarray(rec["base"], bool),
                    clearance=2,
                )
                self.core_total_by_image[name] = int(ms.sum())

    def __len__(self) -> int:
        return len(self.rows)

    def _materialize(self, i: int):
        if i in self._cache:
            return self._cache[i]
        r = self.rows[i]
        record = self.base[r["image"]]
        candidate = compact_row_to_candidate(r)
        sim_score = self.prior.candidate_score(candidate)
        sv, pv, dv, meta = build_relation_views(
            record, candidate, sim_score=sim_score, crop_size=self.crop_size
        )
        y = int(r.get("label", -1))
        g = self.group_map[(r["image"], repr(r["source_key"]))]
        gt = np.asarray(record["gt"], bool)
        base = np.asarray(record["base"], bool)
        add = np.asarray(candidate["add"], bool)
        tp = int((add & gt).sum())
        fp = int((add & ~gt).sum())
        hit, core, _ = core_gap_hit(add, gt, base, clearance=2, tolerance=1)
        metrics = torch.tensor(
            [
                float(tp),
                float(fp),
                float(hit),
                float(core.sum()),
                float(record.get("typ", "") == "normal"),
            ],
            dtype=torch.float32,
        )
        out = (sv, pv, dv, meta, torch.tensor(y), torch.tensor(g), metrics)
        self._cache[i] = out
        return out

    def __getitem__(self, i: int):
        return self._materialize(int(i))


class ComponentBankDataset(Dataset):
    """Conservative Base-component dataset for simulation-aware suppression.

    Labels:
      keep=1 : component overlaps GT by >=2 pixels and >=10% of component area.
      keep=0 : zero GT overlap.
      ambiguous components are excluded.

    This avoids teaching the suppressor to delete weak but real crack fragments.
    """

    def __init__(
        self,
        base_records: dict[str, dict],
        sim_prior: SimulationGeometryPrior,
        *,
        crop_size: int = 33,
    ):
        self.base = base_records
        self.prior = sim_prior
        self.crop_size = int(crop_size)
        self.samples: list[tuple[str, int, int]] = []
        self._cache: dict[int, tuple] = {}
        for name, record in self.base.items():
            b = np.asarray(record["base"], bool)
            gt = np.asarray(record["gt"], bool)
            n, lab = cv2.connectedComponents(b.astype(np.uint8), 8)
            for cid in range(1, n):
                comp = lab == cid
                area = int(comp.sum())
                tp = int((comp & gt).sum())
                if tp == 0:
                    label = 0
                elif tp >= 2 and tp / max(area, 1) >= 0.10:
                    label = 1
                else:
                    continue
                self.samples.append((name, cid, label))
        self.image_names = [s[0] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        if i in self._cache:
            return self._cache[i]
        name, cid, label = self.samples[int(i)]
        record = self.base[name]
        base = np.asarray(record["base"], bool)
        _, lab = cv2.connectedComponents(base.astype(np.uint8), 8)
        comp = lab == cid
        path = _skeleton_diameter_path(comp)
        if len(path) >= 2:
            xy = np.asarray([(x, y) for y, x in path], np.float32)
            sim_score = self.prior.score_polyline(xy)
        else:
            sim_score = 0.5
        view, feat = build_component_view(
            record, comp, sim_score=sim_score, crop_size=self.crop_size
        )
        gt = np.asarray(record["gt"], bool)
        tp = int((comp & gt).sum())
        fp = int((comp & ~gt).sum())
        out = (
            view,
            feat,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(
                [float(tp), float(fp), float(record.get("typ", "") == "normal")],
                dtype=torch.float32,
            ),
        )
        self._cache[i] = out
        return out
