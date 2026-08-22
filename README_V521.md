# CRCV V5.21-dev1 — Compact Dual-Action Corrective Refinement

CRCV V5.21-dev1 is the compact scientific development candidate for corrective crack-segmentation refinement over **frozen Base probability maps**.

The canonical path is deliberately small:

```text
Frozen RGB + Base probability
        |
        v
 nine GT-free shared features
        |
   +----+----+
   |         |
  ADD      REMOVE
   |         |
   +----+----+
        |
  safety projection
        |
        v
 refined mask
```

Training targets are exact natural Base errors:

- `KEEP = Base & GT`
- `REMOVE = Base & ~GT`
- `ADD = GT & ~Base`

GT is used only to construct training targets and development metrics. Runtime takes RGB, a frozen Base probability map, frozen action heads, thresholds, and safety configuration; it has no GT input.

## Canonical scientific choice

V5.21-dev1 promotes the supplied/tested `crcv_core` implementation (core implementation version `1.1.0`) as the canonical development path because it gives a complete bidirectional ADD/REMOVE correction hypothesis while remaining compact and GT-free at runtime.

The previous V5.20.3 counterfactual KEEP/REMOVE implementation remains in `crcv52/` as historical/development evidence and an explicit ablation. It is **not** deleted or relabeled. Counterfactual simulation, V5.4 recovery/proposal machinery, coordinate-only structural banks, and generator-style paths are not part of the V5.21 canonical core.

## Core size

The canonical package contains only:

- `actions.py`
- `features.py`
- `policy.py`
- `safety.py`
- `runtime.py`

The regression contract enforces this five-module scientific surface and a source-size ceiling.

## Development evidence status

The supplied Core v1.1.0 results are development evidence, not final publication evidence. They report positive multi-seed/multi-backbone ADD+REMOVE gains, but the historical development data have been exposed and cannot be reused as an untouched final test.

A fresh ChatGPT-side integration smoke on the user-provided bytes was also run with a tiny CNN Base trained on the debug pack, frozen before CRCV fitting, and evaluated on the provided real-debug split. This is mechanical/development evidence only and must not be cited as paper efficacy.

## Q1 evidence contract

Before a publication claim, freeze the method/configuration and require:

1. a fresh lineage-disjoint final holdout that was not used during development;
2. exact frozen-Base probability/checkpoint provenance;
3. multi-seed paired evaluation;
4. multiple Base backbones and leave-one-backbone-out transfer;
5. source-disjoint cross-dataset routes;
6. a directly relevant published refinement comparator such as StripCuts or an equivalent method;
7. pixel metrics (Precision/Recall/Dice/Crack-IoU) plus topology metrics (clDice/CTS, fragmentation/false-bridge where applicable);
8. per-action ADD/REMOVE precision/recall and mutation-risk metrics such as TCRR/FPRR;
9. fail-closed validation qualification before a corrective policy is allowed to alter final outputs;
10. inference cost and deployment overhead relative to the frozen Base segmentation network.

Null or negative ablations must be reported as such; safety gates must not be relaxed after observing held-out results.

## Local verification

GitHub is source storage, not the smoke-test environment. Before storing this candidate, the canonical core was compiled and its comprehensive standalone regression suite was executed locally in ChatGPT.

```bash
python -m compileall -q crcv_core tests
pytest -q
```

The standalone core implementation version remains `1.1.0`; the repository experiment-family version is `5.21.0.dev1`.
