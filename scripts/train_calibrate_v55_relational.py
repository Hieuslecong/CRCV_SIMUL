#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Sampler

from crcv52.relational_v55 import (
    CRCVV55RelationalBlock,
    recovery_relation_loss,
)
from crcv52.sim_prior import SimulationGeometryPrior
from crcv52.v55_data import RelationBankDataset, load_v54_coregap_bank


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_dataset(model, ds, device, batch_size=256):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores, labels, groups, metrics = [], [], [], []
    for sv, pv, dv, meta, y, g, m in loader:
        out = model.forward_recovery(
            sv.to(device), pv.to(device), dv.to(device), meta.to(device)
        )
        scores.append(torch.sigmoid(out["same_crack_logit"]).cpu().numpy())
        labels.append(y.numpy())
        groups.append(g.numpy())
        metrics.append(m.numpy())
    return (
        np.concatenate(scores),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(groups).astype(np.int64),
        np.concatenate(metrics).astype(np.float64),
    )


def hard_mining_indices(scores, labels, groups, *, hard_negatives=12, no_pos_negatives=3):
    keep = []
    for gid in np.unique(groups):
        ids = np.flatnonzero(groups == gid)
        pos = ids[labels[ids] == 1]
        neg = ids[labels[ids] == 0]
        keep.extend(pos.tolist())
        if len(neg):
            k = hard_negatives if len(pos) else no_pos_negatives
            order = neg[np.argsort(-scores[neg])]
            keep.extend(order[: min(k, len(order))].tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)


class SameSourceBatchSampler(Sampler):
    """Pack complete same-source hard-negative groups into each mini-batch.

    Ranking is undefined if the positive and its hard negatives are silently split
    across unrelated minibatches. This sampler preserves group integrity while
    still shuffling the group order every epoch.
    """
    def __init__(self, indices, group_ids, *, max_candidates=128, seed=5501):
        self.indices = np.asarray(indices, np.int64)
        self.group_ids = np.asarray(group_ids, np.int64)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.epoch = 0
        groups = {}
        for i in self.indices:
            groups.setdefault(int(self.group_ids[i]), []).append(int(i))
        self.groups = list(groups.values())
        if any(len(g) > self.max_candidates for g in self.groups):
            raise ValueError("same-source group exceeds batch candidate budget")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.groups))
        batch = []
        for j in order:
            g = self.groups[int(j)]
            if batch and len(batch) + len(g) > self.max_candidates:
                yield batch
                batch = []
            batch.extend(g)
        if batch:
            yield batch

    def __len__(self):
        n = 0
        cur = 0
        for g in self.groups:
            if cur and cur + len(g) > self.max_candidates:
                n += 1
                cur = 0
            cur += len(g)
        return n + (1 if cur else 0)


def train_epoch(model, ds, indices, optimizer, device, batch_size=128, seed=5501):
    model.train()
    sampler = SameSourceBatchSampler(
        indices, ds.group_ids, max_candidates=batch_size, seed=seed
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    losses = []
    terms = {"rank": [], "same_bce": [], "path_valid": [], "continuity": []}
    for sv, pv, dv, meta, y, g, _ in loader:
        sv, pv, dv, meta = (
            sv.to(device),
            pv.to(device),
            dv.to(device),
            meta.to(device),
        )
        y, g = y.to(device), g.to(device)
        out = model.forward_recovery(sv, pv, dv, meta)
        loss, d = recovery_relation_loss(out, y, g)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        for k in terms:
            terms[k].append(float(d[k]))
    return {
        "loss": float(np.mean(losses)) if losses else None,
        **{k: float(np.mean(v)) if v else None for k, v in terms.items()},
        "n_selected": int(len(indices)),
    }


def candidate_diag(scores, labels):
    if len(np.unique(labels)) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def evaluate_operating_point(
    ds,
    scores,
    metrics,
    *,
    abs_threshold,
    margin_threshold,
    max_accept_per_image=2,
):
    accepted = []
    groups = ds.group_ids
    for gid in np.unique(groups):
        ids = np.flatnonzero(groups == gid)
        order = ids[np.argsort(-scores[ids])]
        top = order[0]
        top_score = float(scores[top])
        second = float(scores[order[1]]) if len(order) > 1 else -np.inf
        margin = top_score - second if np.isfinite(second) else np.inf
        if top_score >= abs_threshold and margin >= margin_threshold:
            accepted.append(top)

    by_image = {}
    for i in accepted:
        by_image.setdefault(ds.image_names[i], []).append(i)
    selected = []
    for name, ids in by_image.items():
        ids = sorted(ids, key=lambda i: float(scores[i]), reverse=True)
        selected.extend(ids[: int(max_accept_per_image)])

    tp = fp = core_hit = normal_added = 0
    details = []
    for name in sorted(set(ds.image_names)):
        ids = [i for i in selected if ds.image_names[i] == name]
        im_tp = int(sum(metrics[i, 0] for i in ids))
        im_fp = int(sum(metrics[i, 1] for i in ids))
        im_hit = int(sum(metrics[i, 2] for i in ids))
        is_normal = bool(
            any(
                metrics[i, 4] > 0.5
                for i in range(len(ds))
                if ds.image_names[i] == name
            )
        )
        if is_normal:
            normal_added += im_tp + im_fp
        else:
            tp += im_tp
            fp += im_fp
            core_hit += im_hit
        details.append(
            {
                "image": name,
                "accepted": len(ids),
                "tp": im_tp,
                "fp": im_fp,
                "core_hit": im_hit,
            }
        )
    core_total = int(sum(ds.core_total_by_image.values()))
    return {
        "absolute_threshold": float(abs_threshold),
        "margin_threshold": float(margin_threshold),
        "max_accept_per_image": int(max_accept_per_image),
        "added_precision": float(tp / (tp + fp + 1e-9)),
        "core_gap_recovery": float(core_hit / (core_total + 1e-9)),
        "tp": int(tp),
        "fp": int(fp),
        "core_hit": int(core_hit),
        "core_total": int(core_total),
        "normal_added": int(normal_added),
        "accepted": int(len(selected)),
        "details": details,
    }


def calibrate(ds, scores, metrics):
    # CAL-only operating point search. Architecture/weights are already frozen.
    abs_grid = np.unique(np.quantile(scores, np.linspace(0.50, 0.999, 80)))
    margin_grid = np.linspace(0.00, 0.50, 21)
    rows = []
    for a in abs_grid:
        for m in margin_grid:
            rows.append(
                evaluate_operating_point(
                    ds,
                    scores,
                    metrics,
                    abs_threshold=float(a),
                    margin_threshold=float(m),
                    max_accept_per_image=2,
                )
            )
    feasible = [
        r
        for r in rows
        if r["added_precision"] >= 0.85
        and r["normal_added"] == 0
        and r["accepted"] > 0
        and r["core_gap_recovery"] >= 0.15
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda r: (
                r["core_gap_recovery"],
                r["added_precision"],
                -r["absolute_threshold"],
            ),
        )
        return best, True
    best = max(
        rows,
        key=lambda r: (
            r["added_precision"],
            r["core_gap_recovery"],
            -r["fp"],
            -r["normal_added"],
        ),
    )
    return best, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--sim-prior",
        type=Path,
        default=Path("artifacts/models/sim_prior_v55.json"),
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5501)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/verifier_v55_relational.pt"),
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/verifier_v55_relational_calibration.json"),
    )
    args = ap.parse_args()
    seed_all(args.seed)

    prior = SimulationGeometryPrior.load(args.sim_prior)
    mbase, mrows = load_v54_coregap_bank(args.artifacts, "module")
    cbase, crows = load_v54_coregap_bank(args.artifacts, "cal")
    train_ds = RelationBankDataset(mbase, mrows, prior)
    cal_ds = RelationBankDataset(cbase, crows, prior)

    device = torch.device(args.device)
    model = CRCVV55RelationalBlock().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params >= 250_000:
        raise RuntimeError(f"V5.5 parameter budget exceeded: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    scores = np.zeros(len(train_ds), np.float32)
    labels = np.asarray([int(r["label"]) for r in train_ds.rows], np.int64)
    groups = train_ds.group_ids
    for epoch in range(args.epochs):
        if epoch > 0:
            scores, labels, groups, _ = score_dataset(
                model, train_ds, device, batch_size=256
            )
        ids = hard_mining_indices(scores, labels, groups)
        stat = train_epoch(
            model,
            train_ds,
            ids,
            opt,
            device,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
        )
        stat["epoch"] = epoch + 1
        history.append(stat)
        print(json.dumps(stat), flush=True)

    # Freeze architecture/weights before CAL. CAL tunes only acceptance thresholds.
    cal_scores, cal_y, _, cal_metrics = score_dataset(model, cal_ds, device)
    diag = candidate_diag(cal_scores, cal_y)
    best, qualified = calibrate(cal_ds, cal_scores, cal_metrics)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "crcv-v5.5-source-path-destination-relational-1",
            "state_dict": model.state_dict(),
            "params": n_params,
            "crop_size": 33,
            "meta_dim": 13,
            "absolute_threshold": best["absolute_threshold"],
            "margin_threshold": best["margin_threshold"],
            "max_accept_per_image": best["max_accept_per_image"],
            "qualified_on_calibration": bool(qualified),
            "seed": args.seed,
        },
        args.checkpoint,
    )
    report = {
        "schema": "crcv-v5.5-relational-calibration-report-1",
        "seed": args.seed,
        "params": n_params,
        "module_candidates": len(train_ds),
        "cal_candidates": len(cal_ds),
        "cal_candidate_diag": diag,
        "calibration": best,
        "qualified": bool(qualified),
        "history": history,
        "final_test": "SEALED_NOT_USED",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "params": n_params,
                "diag": diag,
                "qualified": qualified,
                "calibration": {k: v for k, v in best.items() if k != "details"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
