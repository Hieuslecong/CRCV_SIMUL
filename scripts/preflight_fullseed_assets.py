#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def file_check(path: Path, label: str) -> dict:
    return {"label": label, "path": str(path), "exists": path.is_file(), "kind": "file"}


def dir_check(path: Path, label: str) -> dict:
    return {"label": label, "path": str(path), "exists": path.is_dir(), "kind": "directory"}


def main() -> int:
    p = argparse.ArgumentParser(description="Fail-closed preflight for CRCV full-seed smoke/retraining.")
    p.add_argument("--data-root", default="/mnt/data/v516_data")
    p.add_argument("--proposal-root", default="/mnt/data/v517_work/full/v54_release_full")
    p.add_argument("--v517-module", default="/mnt/data/v519_selfcontained/scripts/fullseed/v517_fullseed.py")
    p.add_argument("--v518-module", default="/mnt/data/v519_selfcontained/scripts/fullseed/v518_fullseed.py")
    p.add_argument("--run-root", default="/mnt/data/v519_fullseed_default")
    args = p.parse_args()

    data = Path(args.data_root)
    proposal = Path(args.proposal_root)
    run_root = Path(args.run_root)

    checks = [
        dir_check(data / "real_debug_data", "real data directory"),
        file_check(data / "real_debug_data" / "manifest.csv", "real-data manifest"),
        dir_check(data / "debug_pack" / "dataset" / "Images", "pretrain images"),
        dir_check(data / "debug_pack" / "dataset" / "Labels", "pretrain labels"),
        file_check(data / "debug_pack" / "dataset" / "train.txt", "pretrain train list"),
        file_check(proposal / "artifacts" / "models" / "geometry_xy.pt", "geometry proposal checkpoint"),
        file_check(proposal / "artifacts" / "models" / "centerline_field_v52b.pt", "centerline-field checkpoint"),
        file_check(proposal / "artifacts" / "models" / "endpoint_ranker_v53.pkl", "endpoint-ranker checkpoint"),
        file_check(Path(args.v517_module), "V5.17 full-seed entrypoint"),
        file_check(Path(args.v518_module), "V5.18 full-seed entrypoint"),
    ]

    # Downstream artifacts are not required before V5.17, but if the user wants
    # to resume directly at V5.18/V5.18.1 these paths explain the dependency.
    downstream = [
        file_check(run_root / "v517" / "v517_banks.pkl", "V5.17 bank for V5.18"),
        file_check(run_root / "v518" / "add_variants_partial.pkl", "V5.18 add variants"),
        file_check(run_root / "v518" / "foreground_authenticity.pkl", "V5.18 foreground authenticity"),
    ]

    missing_required = [x for x in checks if not x["exists"]]
    result = {
        "status": "PASS" if not missing_required else "BLOCKED",
        "required_checks": checks,
        "downstream_resume_checks": downstream,
        "missing_required": [x["label"] for x in missing_required],
        "note": "PASS means the full-seed entrypoint has its minimum input assets. It is not scientific validation.",
    }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
