# Project handoff

## Current bounded objective

Authoritative COCO-32 pose-reference coverage and score-only rescoring for the
completed `overfit32-coco-r64-mse` capacity evaluation are implemented and
CPU-tested. No training, Turbo generation, checkpoint mutation, output
deletion, commit, or push occurred in this session.

## Confirmed completed experiment state

- Training and the seven-checkpoint Turbo generation are complete for the same
  32 COCO training samples (224 generated PNGs total), steps `0, 50, 100, 200,
  300, 400, 500`.
- Checkpoints remain under
  `/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-coco-r64-mse`.
- Evaluation remains under
  `/lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-coco-r64-mse`.
- The old `data/manifests/diagnostic_reference_pose.json` has coverage **0/32**
  for the exact COCO capacity stems. Its null PCK was an evaluation-coverage
  failure, not a training outcome.
- The verified v1 PoseBridge latent coverage for the exact stems is **32/32**.
  The only geometry source is the explicit direct-shard root
  `/lambda/nfs/adhit/krea2-pose/posebridge_latents/train`; only direct
  `train-*.pt` files are considered. Text-conditioning archives, including
  `text_conditioning_v1_backup` duplicate stems, are never geometry sources.

## Implemented exact-reference contract

- `pose_controlnet.reference_pose` resolves the immutable 32-stem manifest,
  reads only direct verified v1 `train-*.pt` shards, preserves persisted
  `source_size`, `resized_size`, `crop_box`, and `bucket` verbatim, and joins
  only official `person_keypoints_train2017.json` / `person_keypoints_val2017.json`.
- It fails closed for bad manifest cardinality, missing/duplicate requested
  stems, malformed shards/geometry, non-COCO manifests, annotation failures,
  output coverage failures, inconsistent geometry, and sidecar integrity or
  provenance mismatches.
- The immutable sidecar path is
  `data/manifests/overfit_capacity_reference_pose/overfit32-coco-r64-mse.jsonl`
  with adjacent `.jsonl.metadata.json`. Metadata records the experiment,
  ordered stems, source manifest SHA-256, explicit latent root/shards, official
  annotation paths/hashes, record/people counts, and records SHA-256.
- Capacity scoring now requires `--reference-sidecar`; it cannot fall back to
  `diagnostic_reference_pose.json`. It validates exact stem coverage and
  sidecar geometry before calling the unchanged authoritative Keypoint R-CNN /
  Hungarian / PCK implementation. Danbooru remains explicitly unavailable
  without real authoritative targets.
- `--stage score-only` consumes existing deterministic generation metadata and
  all existing PNGs, then updates only `training_set_overfit_metrics.json` and
  `overfit_summary.json` (preserving any qualitative-grid references). It does
  not load checkpoints, build a Raw/Turbo model or VAE, sample, train, create
  an optimizer, or call backward.

## Operator commands (foreground; do not run from Codex)

Build the sidecar only after supplying the real official COCO annotation
path(s); use one or both as needed by the 32 stems:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/build_coco_reference_pose.py \
  --experiment overfit32-coco-r64-mse \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents/train \
  --annotations /path/to/person_keypoints_train2017.json /path/to/person_keypoints_val2017.json \
  --output data/manifests/overfit_capacity_reference_pose/overfit32-coco-r64-mse.jsonl
```

Then rescore the already generated 224 PNGs, without generation:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py \
  --experiment overfit32-coco-r64-mse \
  --stage score-only \
  --reference-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-coco-r64-mse.jsonl
```

Expected updated machine-readable artifacts are
`training_set_overfit_metrics.json` and `overfit_summary.json`; existing
`checkpoint_selection_grid.png`, `full_training_set_contact_sheet.png`,
`generation_results.json`, and all 224 generated images are reused unchanged.

## Files changed and checks

- `pose_controlnet/reference_pose.py`
- `scripts/build_coco_reference_pose.py`
- `scripts/evaluate_overfit_capacity.py`
- `tests/test_capacity_reference_pose.py`
- `docs/CODEX_HANDOFF.md`

PASS:

```bash
python -m unittest tests.test_capacity_reference_pose tests.test_reference_pose tests.test_overfit_capacity tests.test_post1500_evaluation -v  # 32 tests
python -m py_compile pose_controlnet/reference_pose.py scripts/build_coco_reference_pose.py scripts/evaluate_overfit_capacity.py tests/test_capacity_reference_pose.py
```

Next action: operator builds the exact immutable sidecar with official COCO
annotations, then runs the foreground score-only command above. Completed
training and generation remain valid and must not be repeated.
