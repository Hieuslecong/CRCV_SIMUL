#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from crcv52.sim_prior import fit_simulation_prior_from_xy


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit CRCV V5.5 geometry-only simulation prior")
    ap.add_argument("xy", type=Path, help="XY trajectory file; literal 0,0 separates trajectories")
    ap.add_argument("--output", type=Path, default=Path("artifacts/models/sim_prior_v55.json"))
    args = ap.parse_args()
    profile = fit_simulation_prior_from_xy(args.xy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_json(args.output)
    print(
        f"saved {args.output} | trajectories={profile.trajectories} "
        f"used={profile.resampled_trajectories} turns={profile.turn_samples}"
    )


if __name__ == "__main__":
    main()
