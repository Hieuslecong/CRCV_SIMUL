# CRCV V5.19-Q1 execution status

## Completed in this update
- Frozen a publication qualification protocol for V5.18.1 rather than adding another architecture module.
- Added hard Q1 readiness gates that fail closed when evidence is missing.
- Added paired image-level bootstrap CI, paired permutation test and Holm correction utilities.
- Added native resolution requirements for 128 and 256.
- Added mandatory five-backbone, three-full-seed, multi-dataset, cross-dataset, LOBO, comparator and latency gates.
- Added six regression tests; all pass locally.

## Current gate
The current evidence intentionally returns BLOCKED. Development gains are strong, but the package refuses to classify them as Q1-ready because 256 completion, three full end-to-end seeds, external datasets, cross-dataset evaluation, V5.18.1 LOBO, an official backbone, paired significance, and CPU/edge latency are still missing.

## Next execution order
1. Finish native 256 for all five backbones and CRCV, using new candidates/ribbon banks at 256.
2. Repeat full Base->CRCV training for seeds 1337/2027/31415 at 256.
3. Add at least two more datasets, with one independent external benchmark; preserve source/lineage separation.
4. Run cross-dataset routes and V5.18.1 LOBO.
5. Add a published refinement comparator and morphology baseline under the same frozen Base masks.
6. Export image-level P/R/F1/mIoU rows; run paired CI/permutation/Holm.
7. Measure CPU and edge latency after method freeze.
8. Freeze config/checkpoints/thresholds, then open final test once.
