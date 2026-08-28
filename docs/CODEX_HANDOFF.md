# Phase 1 handoff

## Current objective and status

Authoritative-v1 -> sidecar-v3 integration remains correctly blocked. A
read-only forensic audit verified the seven invalid Human-Art keypoints and
the apparent final-frame reconstruction failure. No training, dataset or
manifest mutation, coordinate clipping, pose-reward implementation, commit,
or push occurred.

## Current decisions

- `data/pose_targets_authoritative_v1.jsonl` is the only numerical COCO and
  Human-Art source; no original annotation parsing, DWPose, or control-raster
  target extraction is used.
- Sidecar schema is now v3. Active COCO/Human-Art are available only with an
  exact authoritative row; Danbooru remains explicit flow-only unavailable.
- Source points use persisted shard geometry. Visible points beyond declared
  source bounds fail closed; do not relax this validation or silently repair
  / clamp source annotations. Exact far-edge coordinates are accepted then
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
- Active shard and physical RGB/control dimensions exactly match the two
  authoritative rows: painting 1024x589 and sculpture 4128x3096. This is not
  a wrong-dimension/provenance issue.
- `painting_humanart_2000000000804`, annotation `2000000007651`, has five
  visible bottom overshoots: wrist 33.4527 px (5.680% height), knees 246.9159
  / 259.2689 px (41.921% / 44.018%), ankles 606.8460 / 551.7945 px (103.030%
  / 93.683%). These are major annotation-coordinate inconsistencies.
- `sculpture_humanart_14000000001208`, annotation `14000000088574`, has two
  visible bottom overshoots: ankles 81.6055 / 68.4104 px (2.636% / 2.210%
  height). They are material annotation-edge overshoots, not rounding noise;
  strict failure remains appropriate. Both supplied bboxes stay in canvas.
- The requested six reconstruction samples all have source-resolution stored
  controls. At threshold 10, direct final-frame vector re-render IoU is
  0.235–0.665, but actual-control vs vector-source-skeleton passed through the
  exact same PIL preprocessing is 0.674–0.769, with symmetric mean foreground
  distance 0.177–0.303 px and p95 1–2 px. Geometry is consistent.
- Low direct final-frame IoU is a raster-scale comparison error: historical
  3-px source strokes/endpoints are enlarged by LANCZOS, while the vector
  final-frame audit redraws at a fixed 3 px. Report multi-threshold IoU plus
  symmetric distance-transform foreground distance; strict binary IoU alone
  is inadequate after resampling.
- `docs/POSE_TARGET_SIDECAR.md` lines 57–176 are stale legacy v2 material:
  source specification, raw COCO `person_keypoints` JSON, Human-Art adapter
  JSONL, and v2 commands. Do not rewrite it until explicitly asked.

## Files changed this session

- `scripts/diagnose_pose_geometry.py` — reusable read-only diagnostic; writes
  reports/contact sheets only to an explicitly supplied new output directory.
- `docs/CODEX_HANDOFF.md`

## Commands/tests

- PASS: `python -m py_compile scripts/diagnose_pose_geometry.py`
- PASS: diagnostic run for all six requested samples in five bounded output
  runs; JSON reports and six contact sheets under `/tmp/pose_geometry_*_20260828`.
- PASS: direct physical-image inspection confirms RGB/control dimensions for
  both invalid stems match authoritative and shard dimensions.
- EXPECTED FAIL: `scripts/audit_pose_target_sources.py --authoritative-jsonl`
  rejects the first invalid visible source coordinate (painting annotation
  `2000000007651`, joint 9), confirming strict validation remains active.

## Exact next action

Correct and version the authoritative export's seven visible Human-Art
coordinates with upstream evidence, then rerun the authoritative provenance
audit, v3 sidecar build, and reconstruction audit using geometry-aware raster
metrics. Do not implement or enable pose reward until they pass.
