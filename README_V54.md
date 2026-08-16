# CRCV V5.4 — CoreGap Hybrid Proposal

Research status: **proposal qualified, verifier unqualified, runtime fail-closed**.

## Core result
CAL CoreGap oracle: 98.21% exact precision / 32.86% recovery.

Final verifier: not qualified. `CRCVV54Block` therefore returns Base unchanged.

## Tests
```bash
pytest -q
python -m compileall -q crcv52 tests
```

See `artifacts/audit_v54/FINAL_AUDIT_REPORT.md`.
