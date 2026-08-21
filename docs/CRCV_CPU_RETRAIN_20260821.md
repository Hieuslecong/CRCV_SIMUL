# CRCV CPU retraining smoke — 2026-08-21

Branch: `feature/crcv-q1-redesign-v2`

## Decision

**CPU engineering retraining: PASS. Full real-data CRCV retraining: BLOCKED in the active execution runtime because required real-data/upstream assets are not mounted.**

This report must not be used as segmentation-performance evidence. The completed retraining matrix below uses synthetic crack-like records only to verify the exact V5.17 CPU training loop, checkpoint creation, prediction and CAL-style threshold search.

## CPU environment

- PyTorch: `2.10.0+cpu`
- `torch.cuda.is_available() = False`
- CUDA hidden with `CUDA_VISIBLE_DEVICES=''`
- Torch/OMP/MKL threads: 4
- training device: CPU only

## Exact V5.17 training schedule smoke

The training path was extracted from the bundled `v517_fullseed.py` without replacing its model architectures, optimizer, segmentation loss, DataLoader behavior, prediction routine or threshold-search routine.

For every case:

- pretrain epochs: **2**;
- fine-tune epochs: **7**;
- optimizer: AdamW as defined by V5.17;
- five backbones: TinyUNet, FastSCNNLite, BiSeNetTiny, MobileNetV3SmallSeg, DSUNetLite;
- resolutions: 128 and 256;
- seeds: 1337, 2027, 31415.

Matrix size: `5 backbones × 2 resolutions × 3 seeds = 30`.

Result: **30/30 cases completed on CPU and produced checkpoints.**

### Runtime summary

| Resolution | Cases | Total model-train time | Mean train time/case | Median train time/case |
|---|---:|---:|---:|---:|
| 128 | 15 | 10.87 s | 0.725 s | 0.949 s |
| 256 | 15 | 40.39 s | 2.693 s | 2.958 s |

These timings are for a deliberately tiny synthetic smoke dataset and are not representative of full-dataset training cost.

## Synthetic metric warning

The synthetic validation masks are extremely sparse and the smoke dataset is intentionally tiny. Synthetic F1 values were low for several lightweight architectures; these values are **not** a scientific comparison and must not be copied into a paper table.

For audit only:

- mean synthetic F1 @128: 0.02775;
- maximum synthetic F1 @128: 0.10582;
- mean synthetic F1 @256: 0.02048;
- maximum synthetic F1 @256: 0.12491.

The valid conclusion from this section is only that the exact base training loop can complete on CPU for every declared backbone/seed/resolution combination.

## Foreground-authenticity artifact

The separately supplied `foreground_authenticity.pkl` remains compatible with the current V5.18 feature contract:

- SHA256: `5150a08891b9b62d4472509344eb97139dc3c286222c1b0bc30e3712211470b2`;
- LightGBM classifier;
- 13 features;
- 200 estimators;
- `random_state=1856`, consistent with full seed 1337 (`SEED + 519`).

The model can run on CPU and passes the existing 128/256 compatibility smoke. This does not replace retraining from its original FIT records.

## Full real-data CPU attempt

A real V5.17 full-seed entrypoint was invoked with CPU-only environment variables. It cannot reach training in the active runtime because the complete V5.17/V5.18 input tree is not present here.

Current preflight state:

- `foreground_authenticity.pkl`: **present**;
- real-data manifest and raw Images/Labels: **not mounted**;
- debug/pretrain Images/Labels/train list: **not mounted**;
- `geometry_xy.pt`: **not mounted**;
- `centerline_field_v52b.pt`: **not mounted**;
- `endpoint_ranker_v53.pkl`: **not mounted**;
- `v517_banks.pkl`: **not mounted**;
- `add_variants_partial.pkl`: **not mounted**.

Expected V5.4 proposal-checkpoint hashes recorded in the repository are:

- `geometry_xy.pt`: `372b8b0634f08bfca6db39cf89cff21a2aa1f9ac472300505c4dc7166845bf87`;
- `centerline_field_v52b.pt`: `5c43f2dfe734e9cc3e92f6620b0aef45fb3f2504978dc855228f66c9f929c7d4`;
- `endpoint_ranker_v53.pkl`: `426625bd4e798476c75ab9f2f783b233e658330d17f93c81261ea7c6d1ea748c`.

If these files exist on the user's training machine, the blocker is only attachment/mount availability in this execution environment.

## Scientific status

No post-redesign real-data `Delta Dice`, `Delta Crack-IoU`, `Delta F1` or `Delta mIoU` is claimed from this CPU smoke.

The project remains `BLOCKED` for publication evidence until the real-data/upstream assets are available and the full CPU/GPU qualification pipeline is rerun under the frozen protocol.
