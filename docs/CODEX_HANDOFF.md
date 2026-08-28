# Phase 1 handoff

## Current objective and status

Authoritative v1 pose-target sidecar integration is implemented but publication
is correctly blocked by invalid active Human-Art coordinates in the supplied
authoritative export. No training/evaluation run, dataset/manifest mutation,
commit, or push occurred.

## Current decisions

- `data/pose_targets_authoritative_v1.jsonl` is the only numerical COCO and
  Human-Art source; no original annotation parsing, DWPose, or control-raster
  target extraction is used.
- Sidecar schema is now v3. Active COCO/Human-Art are available only with an
  exact authoritative row; Danbooru remains explicit flow-only unavailable.
- Source points use persisted shard geometry. Visible points beyond declared
  source bounds fail closed; exact far-edge coordinates are accepted then
  clipped only in the final frame.
- Reconstruction uses historical body-only unified-18 semantics: renderer-only
  neck, OpenPose rainbow limbs, thickness 3, white radius-4 endpoints.

## Verified findings

- Active coverage by source: COCO 3,893; Danbooru 2,255; Human-Art painting
  6,423; real_human 3,257; sculpture 1,588; total 17,416.
- Exact join check before coordinate validation: all 15,161 active
  COCO/Human-Art stems exist exactly once; 79 Human-Art export rows are
  intentionally inactive; no active Danbooru rows are exported.
- All 21 required annotated diagnostic stems are in the export. Diagnostic
  Danbooru stems `danbooru_anime_11903560`, `danbooru_anime_11910154`, and
  `danbooru_anime_11917323` are absent as required.
- Hard blocker: 7 visible active Human-Art joints lie beyond declared source
  geometry: five on `painting_humanart_2000000000804` and two on
  `sculpture_humanart_14000000001208`. Source dimensions themselves match the
  persisted shard dimensions, so this is an annotation-coordinate/provenance
  inconsistency, not a resize/crop issue.
- Representative renderer comparison (valid `coco_100098_193288`): source
  frame IoU 0.7193 / MAE 0.5429; after persisted resize/crop IoU 0.3010 / MAE
  0.8910. The reconstruction audit now preprocesses stored controls into the
  final training frame; full audit remains blocked until the export is fixed.

## Files changed this session

- `pose_controlnet/pose_targets.py`
- `pose_controlnet/control_reconstruction.py`
- `scripts/build_pose_target_sidecar.py`
- `scripts/audit_pose_target_sources.py`
- `tests/test_pose_targets.py`
- `docs/POSE_TARGET_SIDECAR.md`
- `docs/CODEX_HANDOFF.md`

## Commands/tests

- PASS: `python -m py_compile pose_controlnet/pose_targets.py pose_controlnet/control_reconstruction.py scripts/build_pose_target_sidecar.py scripts/audit_pose_target_sources.py`
- PASS: `python -m unittest tests.test_pose_targets -v` (15 tests).
- EXPECTED FAIL: provenance audit and v3 build stop on the first out-of-range
  active keypoint (`painting_humanart_2000000000804`, joint 9).

## Exact next action

Correct and version the authoritative export's invalid visible Human-Art
coordinates (or correct the associated declared dimensions with evidence),
then rerun the provenance audit, v3 sidecar build, and full reconstruction
audit. Do not implement or enable pose reward until they pass.
