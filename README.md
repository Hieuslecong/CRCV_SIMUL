# CRCV V5.19 — Publication Qualification Redesign

CRCV is a corrective crack-segmentation refinement framework built around two explicit actions over a frozen base prediction:

- **ADD / Recovery**: recover missing crack structures using proposal geometry, RGB evidence and width-aware reconstruction.
- **REMOVE / Suppression**: conservatively remove low-authenticity foreground residuals inside Frozen-Base support.

## Current research status

- Frozen candidate method family: **CRCV V5.18.1**.
- Qualification infrastructure: **V5.19-Q1 v2**.
- Scientific status: **BLOCKED** until independent evidence requirements are satisfied.
- The repository no longer uses `Q1_READY` as a scientific status. The strongest software status is **`EVIDENCE_COMPLETE`**, meaning only that the frozen evidence protocol is complete.

## Non-negotiable publication requirements

1. Historical/development samples may not be relabeled as an untouched final test after they have been exposed during method development.
2. The final external holdout must be fresh, lineage-disjoint and explicitly marked `historically_exposed=false`.
3. Upstream checkpoints and result artifacts must be bound to code/data/configuration through SHA256 provenance.
4. Full training seeds are frozen to `1337`, `2027`, `31415`.
5. At least five backbones are required, including at least two canonical/reference implementations.
6. A published crack-refinement comparator (e.g. StripCuts or an equivalent directly relevant method) is mandatory.
7. LOBO must hold out every declared backbone once.
8. At least two source-disjoint cross-dataset routes are required.
9. Primary endpoints are Dice and crack-foreground IoU; cluster/lineage-aware paired inference is required.
10. A full-pipeline GT mutation test is required before any GT-free runtime claim.

## Core commands

```bash
python -m pip install -e '.[test]'
python -m compileall -q crcv52 crcv_q1 tests
pytest -q
python scripts/q1_readiness.py path/to/evidence.json
```

The evidence validator exits with status `0` only for `EVIDENCE_COMPLETE`; otherwise it fails closed with status `2` and lists missing/invalid evidence.

## Important warning

Existing V5.18/V5.18.1 development gains remain useful development evidence, but they must not be represented as final publication evidence until data-history, checkpoint provenance, full-seed, LOBO, cross-dataset, comparator and statistical gates are satisfied.

See `docs/CRCV_Q1_EXECUTION_PROTOCOL_V2.md` for the execution contract.
