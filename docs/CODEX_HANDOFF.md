# Project handoff

## Current bounded objective

The exact-reference geometry reconciliation bug blocking COCO-32 score-only
evaluation is fixed and CPU-tested. Do not train, generate, run the GH200
evaluation, alter checkpoints, commit, or push as part of this milestone.

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
- The immutable exact COCO sidecar at
  `data/manifests/overfit_capacity_reference_pose/overfit32-coco-r64-mse.jsonl`
  now verifies as **32 records**, **74 people**, and SHA-256
  `2c639f2c671162b711628052cd6f73daa88ed19f3ba001b26816d536e4ab2aef`.
- Its `coco_124949_crowd` record has `source_size=[640,427]`,
  `resized_size=[1247,832]`, `crop_box=[15,0,1231,832]`, and
  `bucket=[1216,832]`.

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
- Root cause: `turbo_scoring_geometry()` required and canonicalized all four
  persisted fields, but returned only `source_size`, `resized_size`, and
  `crop_box`. The strict sidecar loader correctly compares all four, so its
  missing actual `bucket` became `[]` and reconciliation necessarily failed.
- Fix: `turbo_scoring_geometry()` now returns canonical `bucket` along with the
  original three transform fields. The strict sidecar bucket check remains; it
  is not inferred from generated image dimensions. All callers only pass this
  mapping to the unchanged PCK scorer, which consumes the same original three
  transform fields; no training or generation caller uses it.

## Exact next operator action (foreground; do not run from Codex)

Rescore the existing 224 PNGs without generation:

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

## Files changed in this geometry-fix session

- `pose_controlnet/turbo_evaluation.py`
- `tests/test_turbo_evaluation.py`
- `tests/test_post1500_evaluation.py`
- `tests/test_capacity_reference_pose.py`
- `docs/CODEX_HANDOFF.md`

PASS:

```bash
python -m unittest tests.test_turbo_evaluation tests.test_capacity_reference_pose tests.test_post1500_evaluation tests.test_reference_pose tests.test_overfit_capacity tests.test_turbo_lr5e5_evaluation tests.test_turbo_timestep_evaluation -v  # 61 tests
python -m py_compile pose_controlnet/turbo_evaluation.py tests/test_turbo_evaluation.py tests/test_capacity_reference_pose.py tests/test_post1500_evaluation.py
```

Regression coverage proves complete canonical geometry is returned; malformed
and stale geometry fail; exact four-field sidecars reconcile; each
source/resized/crop/bucket mismatch fails closed; bucket addition leaves PCK
results unchanged; and score-only has no generation, checkpoint-loading,
backward, training, or optimizer path.
