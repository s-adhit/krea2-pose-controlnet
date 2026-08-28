# Phase 1 handoff

## Current objective and status

Implemented pre-training, fail-closed authoritative pose-target sidecar and
control-reconstruction infrastructure only. No training, generation, resume,
checkpoint mutation, source-data mutation, network download, commit, or push
occurred. The 1500→1800 continuation remains out of scope.

## Verified facts and provenance audit

- The snapshot has 17,495 physical pairs; active immutable manifests have
  17,416: COCO 3,893; Danbooru 2,255; Human-Art painting 6,423;
  Human-Art real_human 3,257; Human-Art sculpture 1,588. The 79 exclusions
  are Human-Art painting samples.
- Latent shards retain only geometry and latents, not source pose metadata.
  The snapshot has no COCO annotation JSON, Human-Art annotation export,
  historical DWPose JSON/config/checkpoint, or historic renderer source.
  Diagnostic reference annotations cover only 21 examples and are not a
  training target source.
- Therefore all five source families are currently blocked for full-sidecar
  production; the code will not use a control raster or rerun DWPose as a
  substitute. COCO/Human-Art require `original_annotation`; only Danbooru can
  use `dwpose_pseudolabel` with a complete historical provenance record.
- RTMPose-M raw-SimCC is still absent/unvalidated. It remains a future frozen
  reward model only, not target provenance. Existing Keypoint R-CNN/PCK remains
  evaluation-only.

## Completed implementation

- `pose_controlnet/pose_targets.py`: v1 sidecar schema, atomic new-directory
  writer/digest verifier, strict COCO and historical-DWPose readers, source
  size checks, persisted-geometry resize/crop/clipping, per-person boxes,
  confidence/visibility masks, and explicit 17-body-joint mappings. DWPose
  neck is excluded.
- `pose_controlnet/control_reconstruction.py` plus audit script: stratified
  stored-control reconstruction, per-source IoU/MAE mismatch report and
  contact sheet. It refuses an unverified historical renderer.
- `docs/POSE_TARGET_SIDECAR.md` documents the source-spec contract and exact
  build/audit commands. No training modules were changed.

## Commands/tests this session

- PASS: `python -m py_compile ...` for all new modules/scripts.
- PASS: `python -m unittest tests.test_pose_targets -v` (8 tests): resize,
  aspect ratio, crop translation/clipping, partial/multi-person behavior,
  mappings, readonly integrity, fail-closed sources, reconstruction gate.
- PASS (read-only): `python scripts/audit_pose_target_sources.py --dataset-root
  /lambda/nfs/adhit/krea2-pose/posebridge_hf` reports the above exact counts
  and `BLOCKED_MISSING_AUTHORITATIVE_ARTIFACTS`.

## Exact next action

Recover the immutable COCO and Human-Art annotation files plus their exact
stem/image-ID joins, and the historical Danbooru DWPose export/config/model
hashes/renderer source. Put their verified paths and metadata in the documented
source spec, run the source audit, then build and reconstruction-audit v1. Do
not add reward/training code or start training until all sources pass.
