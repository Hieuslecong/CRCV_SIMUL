from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv

FINAL_SPLITS={"final_external","external_test","final_test"}
DEV_SPLITS={"train","fit","cal","val","validation","development"}

def _norm(value): return str(value or "").strip().lower()

def audit_rows(rows:list[dict])->dict:
    failures=[]; warnings=[]; required={"sample_id","split","lineage_id","source_dataset","historically_exposed"}
    ids=set(); by_lineage=defaultdict(set); by_image=defaultdict(set); final_sources=set(); dev_sources=set(); has_final=False
    for i,row in enumerate(rows):
        missing=[k for k in required if k not in row or str(row.get(k,"")).strip()==""]
        if missing: failures.append(f"row {i}: missing {','.join(sorted(missing))}"); continue
        sid=str(row["sample_id"]).strip(); split=_norm(row["split"]); lineage=str(row["lineage_id"]).strip(); source=str(row["source_dataset"]).strip(); exposed=_norm(row["historically_exposed"]) in {"1","true","yes","y"}
        if sid in ids: failures.append(f"duplicate sample_id: {sid}")
        ids.add(sid); by_lineage[lineage].add(split)
        image_sha=str(row.get("image_sha256","")).strip().lower()
        if image_sha: by_image[image_sha].add(split)
        if split in FINAL_SPLITS:
            has_final=True; final_sources.add(source)
            if exposed: failures.append(f"final sample {sid} is historically exposed")
        elif split in DEV_SPLITS: dev_sources.add(source)
    for lineage,splits in by_lineage.items():
        if splits&FINAL_SPLITS and splits&DEV_SPLITS: failures.append(f"lineage leakage: {lineage} appears in development and final splits")
    if has_final:
        for i,row in enumerate(rows):
            if not str(row.get("image_sha256","")).strip(): failures.append(f"row {i}: image_sha256 required when final/external data are declared")
    for digest,splits in by_image.items():
        if len(splits)>1: failures.append(f"exact image leakage across splits: {digest}")
    if final_sources&dev_sources: warnings.append("final and development sets share source_dataset names; source-domain external claims need stronger provenance")
    return {"status":"PASS" if not failures else "FAIL","failures":failures,"warnings":warnings,"n_rows":len(rows),"n_lineages":len(by_lineage),"final_sources":sorted(final_sources),"development_sources":sorted(dev_sources)}

def audit_csv(path:str|Path)->dict:
    with Path(path).open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    return audit_rows(rows)
