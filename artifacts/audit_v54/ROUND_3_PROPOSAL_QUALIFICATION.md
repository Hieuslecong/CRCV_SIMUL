# CRCV V5.4 — Round 3: Proposal qualification

## CAL core-gap oracle
- exact AddedPrecision: **98.21%**
- CoreGap recovery: **32.86%**
- core pixels recovered: **23/70**
- recoverable-image coverage: **4/4 = 100.0%**
- TP/FP additions: **55/1**

This passes the predefined proposal gate of >=95% oracle precision and >=30% core-gap recovery.

## External development oracle (different denominator)
The earlier `real_val` oracle uses regular missing-skeleton semantics rather than CoreGap semantics:
- precision: **98.88%**
- missing-skeleton recovery: **37.45%**
- positive images: **8/10**

These values are supporting diagnostics only and are not numerically compared to the CAL CoreGap metric because the denominators differ.

## Decision
**Proposal stage qualified.** This does not qualify final recovery because oracle uses GT to choose paths.
