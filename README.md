# CRCV V5.21-dev1 — Compact Dual-Action Candidate

> **Branch status:** the canonical scientific development path on `feature/crcv-v5-21-compact-aac-q1` is the five-module natural-error `crcv_core` ADD/REMOVE + AAC implementation. See `README_V521.md`. The V5.20.3 material below is retained as historical/development evidence and an explicit counterfactual REMOVE ablation; it is not deleted or relabeled.

# CRCV V5.20.3 — Counterfactual Corrective Refinement Engineering RC

CRCV is a corrective crack-segmentation refinement framework over frozen Base predictions. The V5.20 research formulation uses explicit corrective actions:

- **ADD**: recover missing crack support.
- **KEEP**: protect already-correct Base foreground.
- **REMOVE**: suppress false-positive Base foreground under fail-closed safety constraints.

The current V5.20.3 engineering release candidate focuses on making the redesigned **KEEP/REMOVE subsystem** deterministic, scale-aware, GT-free at runtime, topology-protected and reproducible. Full redesigned ADD integration remains a scientific milestone rather than a completed claim.

## Current research status

- Current engineering candidate: **V5.20.3**.
- Qualification infrastructure: **V5.19-Q1 v2**, retained as the publication evidence gate.
- Five-pass V5.20.3 engineering review: **PASS** for focused regression/safety/reproducibility checks.
- Multi-seed scientific development gate: **BLOCKED**; 128×128 ACTIVE rates were 4/5 (seed 1337), 2/5 (2027), and 3/5 (31415).
- Publication/Q1 status: **BLOCKED** until the frozen evidence requirements are satisfied.
- `EVIDENCE_COMPLETE` is a software/evidence status only; it is never a claim of journal acceptance.

## V5.20.3 safety/reproducibility contracts

1. Runtime REMOVE APIs do not accept GT.
2. Base skeleton support is protected by the safety projection.
3. Endpoint/junction neighborhoods receive local-width-aware protection.
4. Unsafe/non-beneficial VAL policies fail closed to exact Base behavior.
5. Geometry features and canonical component constraints are normalized for scale; legacy absolute overrides are compatibility-only.
6. Counterfactual additive operators modify only the newly generated primitive, never the whole GT crack.
7. Canonical KEEP/REMOVE training is KEEP-heavy and treats attached FP structures conservatively as IGNORE rather than automatically teaching boundary deletion.
8. FIT record order cannot silently change the V5.20.3 training corpus: record/operator RNG streams are keyed by stable SHA-256 identifiers.
9. Canonical LightGBM training records the feature-schema SHA256 and deterministic training metadata.
10. A separate multi-seed gate prevents a single favorable seed from being treated as scientific robustness.

## Non-negotiable publication requirements

1. Historical/development samples may not be relabeled as an untouched final test after exposure.
2. The final external holdout must be fresh, lineage-disjoint and explicitly certified unexposed.
3. Upstream checkpoints and result artifacts must be bound to code/data/configuration through SHA256 provenance.
4. Full training seeds are frozen to `1337`, `2027`, `31415`.
5. At least five backbones are required, including at least two canonical/reference implementations.
6. A published crack-refinement comparator (e.g. StripCuts or an equivalent directly relevant method) is mandatory.
7. LOBO must hold out every declared backbone once.
8. At least two source-disjoint cross-dataset routes are required.
9. Primary endpoints are Dice and crack-foreground IoU; cluster/lineage-aware paired inference is required.
10. A full-pipeline GT mutation test is required before any GT-free runtime claim.
11. Structural/topology claims require dedicated metrics such as clDice/CTS/fragmentation/false-bridge rate.
12. Full redesigned ADD+KEEP+REMOVE evidence is required before describing V5.20 as a complete bidirectional corrector.

## Core commands

```bash
python -m pip install -e '.[test]'
python -m compileall -q crcv52 crcv_q1 tests
pytest -q
python scripts/q1_readiness.py path/to/evidence.json
```

The evidence validator exits with status `0` only for `EVIDENCE_COMPLETE`; otherwise it fails closed with status `2` and lists missing/invalid evidence.

## Reports

- `docs/CRCV_V5203_FIVE_PASS_REVIEW_20260821.md` — current engineering review and multi-seed development verdict.
- `docs/CRCV_Q1_EXECUTION_PROTOCOL_V2.md` — publication evidence contract.

Existing V5.18/V5.18.1/V5.20.x development gains remain development evidence only and must not be represented as final publication evidence until data-history, checkpoint provenance, full-seed, LOBO, cross-dataset, comparator, structural-metric and statistical gates are satisfied.
