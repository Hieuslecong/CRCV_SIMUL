# CRCV — Crack Relational Correction and Verification

## Active research branch: V5.5

V5.5 implements a **simulation-informed relational correction block** on top of the frozen V5.4 proposal generator.

Current state:

```text
V5.4 proposal            QUALIFIED / FROZEN
V5.5 simulation prior    FITTED
V5.5 relation verifier   IMPLEMENTED / NOT YET QUALIFIED
V5.5 suppression head    IMPLEMENTED / NOT YET QUALIFIED
Recovery runtime         OFF
Suppression runtime      OFF
Final test               SEALED_NOT_USED
```

The V5.4 proposal result remains the current certified recovery upper bound on CAL:

- exact oracle AddedPrecision: **98.21%**
- CoreGapRecovery: **32.86%**

V5.5 uses the supplied simulated-crack XY trajectories only as a **geometry/morphology prior**. Real RGB and natural OOF prediction errors remain mandatory evidence for any keep/remove/connect decision.

The runtime is fail-closed: an unqualified V5.5 head has exactly zero influence and the block returns the frozen Base prediction unchanged.

See:

- `README_V55.md` — V5.5 architecture and reproduction protocol
- `artifacts/audit_v55/IMPLEMENTATION_AUDIT.md` — current V5.5 code/scientific state
- `artifacts/audit_v54/FINAL_AUDIT_REPORT.md` — frozen V5.4 qualification

## Tests

```bash
pytest -q
python -m compileall -q crcv52 tests scripts
```
