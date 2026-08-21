from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv


FINAL_SPLITS = {"final_external", "external_test", "final_test"}
DEV_SPLITS = {"train", "fit", "cal", "val", "validation", "development"}


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def audit_rows(rows: list[dict]) -> dict:
    """Audit split independence using explicit lineage and exposure metadata.

    Required columns/keys for publication manifests:
      sample_id, split, lineage_id, source_dataset, historically_exposed

    `historically_exposed` must be false for every final/external sample. A final
    holdout is invalid if its lineage occurs in any development split.
    """
    failures: list[str] = []
    warnings: list[str] = []
    required = {"sample_id", "split", "lineage_id", "source_dataset", "historically_exposed"}

    ids: set[str] = set()
    by_lineage: dict[str, set[str]] = defaultdict(set)
    final_sources: set[str] = set()
    dev_sources: set[str] = set()

    for i, row in enumerate(rows):
        missing = [k for k in required if k not in row or str(row.get(k, "")).strip() == ""]
        if missing:
            failures.append(f"row {i}: missing {','.join(sorted(missing))}")
            continue

        sid = str(row["sample_id"]).strip()
        split = _norm(row["split"])
        lineage = str(row["lineage_id"]).strip()
        source = str(row["source_dataset"]).strip()
        exposed = _norm(row["historically_exposed"]) in {"1", "true", "yes", "y"}

        if sid in ids:
            failures.append(f"duplicate sample_id: {sid}")
        ids.add(sid)
        by_lineage[lineage].add(split)

        if split in FINAL_SPLITS:
            final_sources.add(source)
            if exposed:
                failures.append(f"final sample {sid} is historically exposed")
        elif split in DEV_SPLITS:
            dev_sources.add(source)

    for lineage, splits in by_lineage.items():
        if splits & FINAL_SPLITS and splits & DEV_SPLITS:
            failures.append(
                f"lineage leakage: {lineage} appears in development and final splits"
            )

    if final_sources & dev_sources:
        warnings.append(
            "final and development sets share source_dataset names; this may be lineage-disjoint "
            "but must not be described as source-domain external without stronger provenance"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "n_rows": len(rows),
        "n_lineages": len(by_lineage),
        "final_sources": sorted(final_sources),
        "development_sources": sorted(dev_sources),
    }


def audit_csv(path: str | Path) -> dict:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return audit_rows(rows)
