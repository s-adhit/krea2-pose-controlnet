# Phase 1 handoff

## Current objective

Gate B physical dataset resolution is implemented and verified. The next
bounded objective is paired preprocessing using this shared index. Do not begin
VAE encoding, latent-shard creation, model download, or training without a
separate assignment.

## Decisions in force

- Krea-2 Raw, rendered skeleton control, spatial channel concatenation,
  rank-64 LoRA, BF16, and flow-matching MSE only remain unchanged.
- Source files and immutable manifests are read-only. Manifest membership,
  never Hugging Face storage paths, defines each split.
- `pose_controlnet.dataset_index` is the project-owned shared physical resolver
  for preprocessing, training, verification, and evaluation.

## Verified environment facts

- Host verification remains: Linux aarch64, GH200, Python 3.10.12, torch
  2.7.0+cu128, CUDA runtime 12.8, cuDNN 9.8, Triton 3.3.0, and `uv 0.12.5`.
- This audit shell has no CUDA. Gate A CUDA checks and the prior DNS/`uv lock`
  issue remain outside this session's scope.

## Gate B — PASS

- Read-only snapshot: `/home/ubuntu/data/posebridge_hf`.
- Physical layout: 34,995 non-cache payload files; recursive RGB discovery at
  `images/**/*.jpg` and controls at `conditioning_images/**/*.png`, stored
  under `shard_00` through `shard_08`.
- Metadata and immutable manifests: `metadata.jsonl`, `manifests/train.jsonl`,
  `manifests/val.jsonl`, and `manifests/diagnostic_val.jsonl`.
- PASS: 17,495 unique RGB stems and 17,495 unique control stems; sets exactly
  equal, so every physical pair is unambiguous.
- PASS: manifests resolve 16,503 train, 889 representative val, and 24
  diagnostic-val records (17,416 total); splits are disjoint, every record
  resolves to exactly one RGB/control pair, and every caption is non-empty.
- The index fails loudly for duplicate stems, missing counterparts, malformed
  or non-bare manifest filenames, unresolved records, duplicate/overlapping
  split records, and empty captions.

## Exact checks

- `python -m unittest -v tests/test_dataset_index.py` — PASS (5 tests):
  recursive discovery, duplicate detection, missing counterpart detection,
  manifest resolution/schema/caption checks, and split disjointness.
- `python -m pose_controlnet.dataset_index --dataset-root /home/ubuntu/data/posebridge_hf` — PASS; reports required physical and manifest counts.
- `pytest -q tests/test_dataset_index.py` — NOT RUN: `pytest` is not installed
  in this audit shell; equivalent standard-library tests passed without
  changing the environment.
- `git diff --check` — PASS.
- `git status --short` — `M docs/CODEX_HANDOFF.md`, plus untracked
  `pose_controlnet/dataset_index.py` and `tests/test_dataset_index.py`.

## Files changed this session

- `pose_controlnet/dataset_index.py`
- `tests/test_dataset_index.py`
- `docs/CODEX_HANDOFF.md`

## Current blockers

None for the completed physical-index milestone. Gate A's CUDA and network
conditions remain recorded above and are not superseded by this CPU-only work.

## Exact next recommended action

Implement paired preprocessing using `DatasetIndex.discover()` and
`validate_manifests()` as the only physical path-resolution source. Preserve
manifest-defined membership and apply identical bucket/resize/crop geometry to
the resolved RGB and control files.
