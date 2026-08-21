#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

from crcv_q1.gates import assess_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a CRCV publication evidence package."
    )
    parser.add_argument("evidence_json", help="Path to evidence package JSON")
    parser.add_argument(
        "--root",
        default=None,
        help="Artifact root. Defaults to the evidence JSON directory.",
    )
    args = parser.parse_args()
    result = assess_file(args.evidence_json, root=args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "EVIDENCE_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
