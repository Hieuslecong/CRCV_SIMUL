# CRCV V5.4 — Round 2: Proposal-family redesign

## Active families
1. V5.2b multi-target field/geodesic.
2. V5.2b top-K spatial target hypotheses with NMS.
3. V5.3 iterative tracer.
4. V5.4 RGB ridge open-CONTINUE walker.

## Key fix
Long open tails such as `c1228` were not recoverable by the fixed geometry corridor because the XY rollout drifted at long horizon. The new RGB-ridge CONTINUE walker treats geometry as a local prior rather than a long-horizon hard corridor and produced exact-positive proposals for `c1228`, `c1093`, `c1316`, and other natural gaps.

## Rejected CONNECT families
- destination-conditioned Dijkstra CONNECT: produced shortcut/off-GT paths;
- bidirectional ridge meet-in-the-middle CONNECT: produced no useful CAL positive paths.

Both are removed from the active proposal set rather than retained for architectural appearance.
