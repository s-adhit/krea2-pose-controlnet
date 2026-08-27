# Phase 1 handoff

## Current bounded objective and status

Turbo benchmark PCK scoring geometry is repaired. This is an evaluation-only
metadata fix: no training, optimizer, model/schedule change, checkpoint change,
or Turbo image generation was performed. The existing Turbo images under
`/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0/fixed_pose/` remain
the inputs for the retry and **do not need to be regenerated**.

## Root cause and exact fix

- `scripts/turbo_benchmark.py score` built `geometry_by_stem` from
  `PreparedLatentShardDataset.__getitem__()` values. That view exposed latents,
  caption, and stem, but dropped the prepared shard's persisted
  `source_size`, `resized_size`, and `crop_box`; PCK therefore raised
  `KeyError: 'source_size'`.
- `PreparedLatentShardDataset` now preserves `bucket`, `source_size`,
  `resized_size`, and `crop_box` for read-only evaluators; training collation
  continues to ignore metadata.
- `turbo_scoring_geometry()` consumes the persisted source size/bucket and
  reuses `resize_center_crop_geometry()` from project-owned paired
  preprocessing. It verifies the persisted resize/crop values equal the
  canonical contract (resize-to-cover using `round`, then integer center crop)
  and returns the exact three fields required by `reference_people_from_sidecar`.
  It never derives geometry from generated image pixels.
- `score_authoritative_pck()` now reports missing geometry fields with a clear
  `ValueError` instead of leaking a `KeyError`.

## Evaluation rules still in force

- Turbo remains Krea-2 Turbo, 8 steps, CFG 0, mu 1.15; no Turbo schedule/model
  loading behavior changed.
- Authoritative PCK semantics are unchanged: 21 Human-Art/COCO samples, three
  unavailable Danbooru records excluded, source-visible AND rendered-control
  joints, COCO-17 Keypoint R-CNN COCO_V1 at confidence >= 0.5, deterministic
  Hungarian association, bbox-diagonal normalization, `<=` thresholds, and
  unmatched reference people in denominators/failures.
- Outputs remain isolated at
  `/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0`.

## Files changed this session

- `pose_controlnet/data.py`
- `pose_controlnet/turbo_evaluation.py`
- `scripts/turbo_benchmark.py`
- `pose_controlnet/post1500_evaluation.py`
- `tests/test_turbo_evaluation.py`
- `tests/test_post1500_evaluation.py`

## Verified tests

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_turbo_evaluation tests.test_post1500_evaluation tests.test_reference_pose` (27 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile scripts/turbo_benchmark.py pose_controlnet/turbo_evaluation.py pose_controlnet/post1500_evaluation.py tests/test_turbo_evaluation.py`.
- PASS: `git diff --check`.
- Regression coverage proves persisted Turbo geometry includes all required
  fields, matches paired preprocessing for portrait/landscape/square examples,
  saved Turbo outputs score without calling generation, missing fields fail
  clearly, and canonical PCK pooling/exclusion semantics remain unchanged.

## Exact GH200 retry (do not train or regenerate)

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
cd /home/ubuntu/Krea-2-Pose-ControlNet
uv run python scripts/turbo_benchmark.py score --turbo-ckpt "$OSS_TURBO"
uv run python scripts/turbo_benchmark.py report --turbo-ckpt "$OSS_TURBO"
```
