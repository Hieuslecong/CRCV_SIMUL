# CRCV V5.5 — Simulation-Informed Relational Correction Block

## Status

**IMPLEMENTED / NOT YET QUALIFIED / FAIL-CLOSED**

V5.5 is deliberately built on the frozen V5.4 proposal bank. It does not reopen geometry/proposal tuning. The final test remains sealed.

## Core idea

The same simulated-crack source contributes two *priors*, not two decisions:

1. **Simulation morphology prior** — describes plausible crack geometry and is exposed as a weak feature for conservative false-component suppression.
2. **Simulation propagation prior** — ranks geometrically plausible continuations and is exposed to the recovery verifier.

Neither prior is allowed to modify the segmentation by itself. Runtime correction requires real RGB evidence and natural OOF errors.

```text
RGB + frozen Base
        |
        +-------------------------------+
        |                               |
  Base components                 Frozen V5.4 proposals
        |                               |
 simulation morphology            simulation propagation
      prior feature                    prior feature
        |                               |
 component RGB keep head     source-path-destination relation head
        |                               |
 conservative suppression      Top-1 + margin + abstention
        +---------------+---------------+
                        |
                Structural Safety Gate
                        |
                  refined mask
```

## Recovery head

`CRCVV55RelationalBlock` uses one **64,684-parameter** shared depthwise encoder for three local views:

- source context,
- candidate-path context,
- destination/future context.

Each view has 8 channels:

`RGB(3) + BaseProb + BaseMask + CandidateMask + RGB ridge + role mask`.

The simulation score is kept as a scalar meta feature so a geometry prior cannot masquerade as direct image evidence.

Relation fusion uses source/path/destination embeddings, absolute embedding differences, embedding products and candidate metadata. It predicts:

- `same_crack_logit`,
- `path_valid_logit`,
- `continuity_logit`.

Training uses **same-source hard-negative ranking + focal BCE + auxiliary path/continuity**. Same-source candidates are kept in the same mini-batch so the ranking term is never silently neutralized by ordinary random batching.

At calibration, a candidate is accepted only when it is:

1. top-1 within the same source group,
2. above an absolute score threshold,
3. separated from the second-best path by a minimum margin.

Otherwise the block **abstains**.

## Suppression head

Suppression is component-level, not free pixel deletion. A component crop reuses the same 8-channel image encoder and adds a 12-D structural vector containing morphology, Base confidence, RGB ridge support, skeleton statistics and simulation plausibility.

Training labels are deliberately conservative:

- drop candidate: zero GT overlap,
- keep candidate: >=2 GT pixels and >=10% component overlap,
- uncertain components: excluded.

False removal of real crack pixels receives a large asymmetric loss weight. Even if the CAL suppression gate passes, runtime suppression remains disabled until multi-backbone validation is completed.

## Simulation prior

`crcv52/sim_prior.py` reads XY trajectories separated by literal `0,0`. It does **not** interpret blank lines as simulation families.

Before morphology statistics are estimated, each polyline is arc-length resampled to remove duplicate/nearly duplicate simulation points. This prevents mesh/sampling artefacts from dominating turning-angle statistics.

The committed profile was fitted from the supplied XY file:

- 761 valid trajectories under the existing `parse_xy` minimum-length rule,
- 20,849 resampled turning samples,
- geometry only: no RGB, no crack width, no material label.

The raw simulation file is intentionally not committed.

## Qualification gates

### Recovery

- AddedPrecision >= 0.85; target >= 0.90
- CoreGapRecovery >= 0.15 after the verifier
- NormalAdded = 0
- clDice must improve over the same frozen Base
- CC error must not worsen

### Suppression

- TruePixelRemoval <= 0.01
- False-pixel removal >= 0.30
- multi-backbone validation before runtime enablement

### Runtime

Unqualified heads have exactly zero influence. `config_v55.json` currently keeps:

```text
recovery_enabled    = false
suppression_enabled = false
width_enabled       = false
joint_training      = false
runtime_policy      = FAIL_CLOSED_BASE_ONLY
final_test          = SEALED_NOT_USED
```

## Reproduction sequence

```bash
python scripts/fit_v55_sim_prior.py data/simulation/crack_xy.txt

python scripts/train_calibrate_v55_relational.py \
  --artifacts artifacts \
  --device cuda

# Only after the recovery representation is frozen:
python scripts/train_calibrate_v55_suppression.py \
  --artifacts artifacts \
  --device cuda
```

The V5.4 Module-TRAIN/CAL cache is intentionally not stored in GitHub. Training therefore must be run in the certified research workspace that contains the frozen OOF banks.

## What V5.5 does not claim yet

- no final-test performance,
- no deployable recovery improvement yet,
- no qualified suppression yet,
- no width reconstruction,
- no Base+CRCV joint training,
- no Q1 claim based on implementation alone.
