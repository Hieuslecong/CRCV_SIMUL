# CRCV V5.4 — Round 1: Target semantics and source audit

## Finding
The previous `missing GT skeleton = GT skeleton AND NOT Base` target mixed true topological gaps with 1–2 px Base/GT centerline misregistration. On CAL, 42.5% of the nominal missing skeleton was within 2 px of Base. Treating those pixels as continuation targets teaches the recovery block to draw a parallel centerline beside an already-present crack.

## Fix
V5.4 defines the recovery target as:

`CoreGap = Skeleton(GT) AND NOT Dilate(Base, radius=2)`

This changes only the topology-recovery denominator/labels. Exact `AddedPrecision` remains exact pixel overlap with the GT mask; it is not relaxed.

## Source fix
Source selection is component-balanced (max 2 endpoints/component) with border artifact filtering, preventing one noisy component from consuming the entire source budget.

## Decision
PASS correctness/protocol closure for V5.4 recovery semantics. Final test remains sealed.
