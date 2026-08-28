# Phase 1 handoff

## Current objective and status

Pose-target sidecars now support partial authoritative pose-reward coverage.
No training, resume, generation, checkpoint mutation, dataset/manifest
mutation, download, commit, or push occurred in this session.

## Current decisions

- First ControlNet++ pose-reward experiment: Human-Art and COCO claim
  `pose_reward_available: true` with `target_provenance: original_annotation`.
- Danbooru is intentionally flow-only: `pose_reward_available: false`,
  `target_provenance: unavailable`, `format: unavailable`. It does not block a
  sidecar build, coverage audit, or reconstruction audit.
- The sidecar schema is v2. Available and unavailable records are explicit;
  the training-facing `pose_reward_target_for_stem` returns a target record or
  `None` only for an explicit unavailable record.
- There is no raster-to-keypoint path and no DWPose/pseudo-label/rerun fallback
  in the pose-target builder.
- Human-Art must first be converted from the user-supplied raw schema into the
  documented canonical JSONL adapter. Original source parsing remains outside
  the common sidecar representation.
- COCO uses its original `person_keypoints_train2017.json` annotations and the
  immutable `coco_<image_id>_<annotation_id|crowd>` stem join.

## Completed/green gates

- v2 records and metadata report exact total/available/unavailable coverage
  counts and percentages per source and overall.
- Claimed available targets fail closed for missing records, malformed COCO
  keypoints, source-size mismatch, non-finite/negative keypoints, and visible
  points outside source geometry.
- Reconstruction selection considers only available records and adds complete
  coverage to its output; unavailable Danbooru is not a reconstruction failure.
- PASS: `python -m py_compile pose_controlnet/pose_targets.py
  pose_controlnet/control_reconstruction.py scripts/build_pose_target_sidecar.py
  scripts/audit_pose_target_sources.py scripts/audit_control_reconstruction.py
  tests/test_pose_targets.py`.
- PASS: `python -m unittest tests.test_pose_targets -v` (12 tests).
- PASS: `git diff --check`.

## Files changed this session

- `pose_controlnet/pose_targets.py`
- `pose_controlnet/control_reconstruction.py`
- `scripts/audit_pose_target_sources.py`
- `scripts/audit_control_reconstruction.py`
- `tests/test_pose_targets.py`
- `docs/POSE_TARGET_SIDECAR.md`
- `docs/CODEX_HANDOFF.md`

## Exact next action

When the user supplies the Human-Art annotation export and original COCO JSON,
create the canonical Human-Art adapter and source spec following
`docs/POSE_TARGET_SIDECAR.md`, then run:

```bash
python scripts/audit_pose_target_sources.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --source-spec /absolute/path/to/pose_reward_source_spec.json
```

If it passes, build v2 and run the available-only reconstruction audit. Do not
start or resume training as part of that work unless separately authorized.
