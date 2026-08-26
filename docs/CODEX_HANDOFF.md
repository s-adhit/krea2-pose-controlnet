# Phase 1 handoff

## Current objective

Gate B physical resolution and paired geometric preprocessing are implemented
and verified. This session intentionally did not begin VAE encoding,
latent-shard creation, model download, text encoding, or training.

## Decisions in force

- Krea-2 Raw, rendered skeleton control, spatial channel concatenation,
  rank-64 LoRA, BF16, and flow-matching MSE only remain unchanged.
- Source files and immutable manifests are read-only. Manifest membership,
  never Hugging Face storage paths, defines each split.
- `pose_controlnet.dataset_index` is the sole project-owned physical resolver.
- `pose_controlnet.paired_preprocessing` owns the one shared paired geometry:
  nearest aspect ratio in log space over the fixed nine Krea buckets;
  resize-to-cover using `round`; then floor-offset center crop.
- A pair must have identical source dimensions before it is transformed. RGB
  and control receive the same selected bucket, scale, resize dimensions, and
  crop box; files are only opened/read by this milestone.

## Verified environment facts

- Host verification remains: Linux aarch64, GH200, Python 3.10.12, torch
  2.7.0+cu128, CUDA runtime 12.8, cuDNN 9.8, Triton 3.3.0, and `uv 0.12.5`.
- This audit shell has no CUDA and provides Pillow 9.0.1; paired preprocessing
  uses the compatible Lanczos constant while preserving the same filter.

## Gate B — PASS

- Read-only snapshot: `/home/ubuntu/data/posebridge_hf`.
- PASS: 17,495 unique RGB stems and 17,495 unique control stems; physical sets
  match exactly. Immutable manifests resolve 16,503 train, 889 val, and 24
  diagnostic-val entries (17,416 total), with disjoint splits and non-empty
  captions.
- The index fails loudly for duplicate/missing counterparts, malformed names,
  unresolved records, duplicate/overlapping splits, and empty captions.

## Gate C paired geometry — PASS

- Fixed buckets: 1024x1024, 896x1152, 1152x896, 832x1216, 1216x832,
  768x1344, 1344x768, 704x1472, 1472x704.
- Tests cover square/portrait/landscape and extreme aspect ratios, deterministic
  crop coordinates, exact dimensions, shared pair geometry, source-size
  mismatch, malformed images, and missing physical pairs.
- Read-only inspection CLI: `python -m pose_controlnet.paired_preprocessing
  --dataset-root /home/ubuntu/data/posebridge_hf --split train --limit 3`.
  It first validates all immutable manifest memberships through `DatasetIndex`,
  then prints only selected sample summaries. The three checked train samples
  produced valid 1216x832 or 1152x896 outputs with their shared geometry.

## Exact checks

- `python -m unittest -v tests/test_dataset_index.py tests/test_paired_preprocessing.py` — PASS (13 tests).
- `python -m pose_controlnet.paired_preprocessing --dataset-root /home/ubuntu/data/posebridge_hf --split train --limit 3` — PASS.
- `git diff --check` — PASS after this handoff rewrite.
- `git status --short` — `M docs/CODEX_HANDOFF.md`, plus untracked
  `pose_controlnet/paired_preprocessing.py` and
  `tests/test_paired_preprocessing.py`.

## Files changed this session

- `pose_controlnet/paired_preprocessing.py`
- `tests/test_paired_preprocessing.py`
- `docs/CODEX_HANDOFF.md`

## Current blockers

None for Gate B and paired geometric preprocessing. Gate A CUDA/network
conditions remain outside this CPU-only milestone.

## Exact next recommended action

Await a separately bounded assignment for Gate C VAE preprocessing and latent
shard creation. It must consume `ManifestRecord` paths from `DatasetIndex` and
the shared output of `paired_preprocessing`; do not reimplement path resolution
or paired crop geometry.
