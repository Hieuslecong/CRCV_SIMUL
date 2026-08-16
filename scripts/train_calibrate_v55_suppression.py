#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from crcv52.relational_v55 import CRCVV55RelationalBlock, suppression_safety_loss
from crcv52.sim_prior import SimulationGeometryPrior
from crcv52.v55_data import ComponentBankDataset, load_v54_coregap_bank


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_components(model, ds, device, batch_size=256):
    model.eval()
    probs, labels, metrics = [], [], []
    for view, feat, y, m in DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        logit = model.forward_component(view.to(device), feat.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
        metrics.append(m.numpy())
    return (
        np.concatenate(probs) if probs else np.empty(0),
        np.concatenate(labels).astype(np.int64) if labels else np.empty(0, np.int64),
        np.concatenate(metrics) if metrics else np.empty((0, 3)),
    )


def calibrate_suppression(probs, metrics):
    # keep_prob <= threshold means remove whole component.
    if not len(probs):
        raise RuntimeError("empty CAL component set")
    total_true = float(metrics[:, 0].sum())
    total_false = float(metrics[:, 1].sum())
    grid = np.unique(np.quantile(probs, np.linspace(0.01, 0.75, 120)))
    rows = []
    for t in grid:
        rem = probs <= t
        true_removed = float(metrics[rem, 0].sum())
        false_removed = float(metrics[rem, 1].sum())
        normal_removed = float((metrics[rem, 1] * metrics[rem, 2]).sum())
        tp_rate = true_removed / (total_true + 1e-9)
        fp_rate = false_removed / (total_false + 1e-9)
        rows.append(
            {
                "keep_threshold": float(t),
                "true_pixel_removal": float(tp_rate),
                "false_pixel_removal": float(fp_rate),
                "true_removed": int(true_removed),
                "false_removed": int(false_removed),
                "normal_fp_removed": int(normal_removed),
                "components_removed": int(rem.sum()),
            }
        )
    feasible = [
        r
        for r in rows
        if r["true_pixel_removal"] <= 0.01
        and r["false_pixel_removal"] >= 0.30
        and r["components_removed"] > 0
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda r: (
                r["false_pixel_removal"],
                -r["true_pixel_removal"],
                -r["keep_threshold"],
            ),
        )
        return best, True
    best = max(
        rows,
        key=lambda r: (
            -r["true_pixel_removal"],
            r["false_pixel_removal"],
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
    ap.add_argument(
        "--relation-checkpoint",
        type=Path,
        default=Path("artifacts/models/verifier_v55_relational.pt"),
        help="Frozen V5.5 recovery checkpoint; its encoder is frozen for suppression.",
    )
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5511)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/suppress_v55.pt"),
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/suppress_v55_calibration.json"),
    )
    args = ap.parse_args()
    seed_all(args.seed)

    if not args.relation_checkpoint.exists():
        raise FileNotFoundError(
            "Train/freeze V5.5 relation verifier first. Suppression may not alter its encoder."
        )
    prior = SimulationGeometryPrior.load(args.sim_prior)
    mbase, _ = load_v54_coregap_bank(args.artifacts, "module")
    cbase, _ = load_v54_coregap_bank(args.artifacts, "cal")
    train_ds = ComponentBankDataset(mbase, prior)
    cal_ds = ComponentBankDataset(cbase, prior)

    device = torch.device(args.device)
    ck = torch.load(args.relation_checkpoint, map_location="cpu")
    model = CRCVV55RelationalBlock()
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device)

    # Recovery representation is frozen. Only the component head learns.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.component_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        model.component_head.parameters(), lr=1e-3, weight_decay=1e-4
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for view, feat, y, _ in DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ):
            logit = model.forward_component(view.to(device), feat.to(device))
            loss = suppression_safety_loss(
                logit, y.to(device), true_crack_weight=8.0
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.component_head.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        stat = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else None,
        }
        history.append(stat)
        print(json.dumps(stat), flush=True)

    probs, labels, metrics = score_components(model, cal_ds, device)
    best, cal_qualified = calibrate_suppression(probs, metrics)

    # CAL passing is necessary but not sufficient: require multi-backbone validation
    # before any suppression can influence deployed output.
    runtime_qualified = False
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "crcv-v5.5-simulation-aware-component-keep-1",
            "state_dict": model.state_dict(),
            "keep_threshold": best["keep_threshold"],
            "calibration_gate_pass": bool(cal_qualified),
            "runtime_qualified": runtime_qualified,
            "encoder_frozen_from": str(args.relation_checkpoint),
            "seed": args.seed,
        },
        args.checkpoint,
    )
    report = {
        "schema": "crcv-v5.5-suppression-calibration-report-1",
        "seed": args.seed,
        "module_components": len(train_ds),
        "cal_components": len(cal_ds),
        "calibration": best,
        "calibration_gate_pass": bool(cal_qualified),
        "runtime_qualified": runtime_qualified,
        "runtime_blocker": "multi-backbone suppression validation not yet executed",
        "history": history,
        "final_test": "SEALED_NOT_USED",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
