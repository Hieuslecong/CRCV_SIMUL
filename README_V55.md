# CRCV V5.5+ — Simulation-Informed Correction Research Branch

## Current status

**RESEARCH ITERATION / FAIL-CLOSED.** V5.4 remains the frozen proposal baseline. The final test remains sealed.

Current evidence:

- V5.4 proposal oracle: **QUALIFIED / FROZEN**.
- Simulation geometry prior: retained for recovery proposal/trajectory geometry only.
- Simulation as an active suppression signal: **NOT SUPPORTED** after the GT-free V5.5c re-audit.
- GT-free component-confidence suppression on Original-V52: CAL positive signal, but **not multi-backbone qualified** and remains OFF.
- V5.5/V5.6 recovery verifiers: FAIL.
- V5.7 ordered path-aligned strip verifier: FAIL.
- V5.7c full 3-fold OOF verifier training: FAIL.
- V5.8 six-fold closer-OOF verifier training: FAIL.
- Recovery: OFF.
- Suppression: OFF.
- Runtime: `FAIL_CLOSED_BASE_ONLY`.
- Final test: `SEALED_NOT_USED`.

## Simulation role

The XY trajectory source remains a **geometry prior**, not direct image evidence. It can encode plausible turning/tortuosity/propagation behavior for recovery proposals, but it cannot authorize a connection by itself. Real RGB and natural upstream errors are required for acceptance.

The earlier hypothesis that the same simulation prior should actively improve suppression was tested. After correcting a P0 runtime-candidate-space bug, simulation did not beat the much simpler Base component-confidence rule; therefore that suppression claim is rejected unless future clean experiments provide new evidence.

## Recovery experiments

V5.7 replaces square-resized path crops with ordered, path-aligned cross-sections containing RGB, Base probability/mask, ridge/gradient evidence and trajectory position. V5.7b adds GT-free topology evidence such as distance to Base and source-tangent continuity.

Full OOF experiments then expanded natural verifier-training errors. A 3-fold OOF construction produced many more positive source events but still failed transfer to CAL. V5.8 repeated the experiment with 6-fold OOF models trained on 30/36 Base-FIT records to reduce upstream mismatch; recovery performance became worse, not better.

The current audit therefore treats **upstream covariate shift and matched natural-error scarcity** as the main research blocker. Do not stack another recovery model without first resolving that protocol/data-distribution problem.

## Qualification gates

Recovery may only be enabled after all required gates pass, including at least:

- AddedPrecision >= 0.85,
- CoreGapRecovery >= 0.15,
- NormalAdded = 0,
- clDice improvement over the same frozen Base,
- CC error not worse than Base.

Suppression may only be enabled after conservative true-pixel removal and false-pixel-removal gates pass on properly trained frozen backbones using a fresh validation protocol.

## Authoritative audits

- `artifacts/audit_v55/P0_SUPPRESSION_RUNTIME_REAUDIT.md`
- `artifacts/audit_v55/V56_RECOVERY_REDESIGN_TRIAL.md`
- `artifacts/audit_v55/V58_PATH_STRIP_OOF_SHIFT_AUDIT.md`

Unqualified modules have exactly zero runtime influence.
