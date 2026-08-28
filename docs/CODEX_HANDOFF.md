# Phase 1 handoff

## Current objective and status

Authoritative-v1 -> sidecar-v3 provenance semantics are implemented and
verified. The exact seven original Human-Art visible source-out-of-bounds
joints are faithfully preserved for historical reconstruction and masked from
future pose-reward use. No pose reward was implemented, no targets/manifests
or source data changed, and no training, commit, or push occurred.

## Current decisions

- `data/pose_targets_authoritative_v1.jsonl` remains the sole numerical source
  for active COCO/Human-Art targets. Danbooru remains explicit flow-only
  `pose_reward_available=false`.
- Sidecar schema remains v3. Every available joint now has raw source point,
  visibility/confidence, source-in-bounds, unclipped training coordinate,
  final-frame status, reward validity, and invalid reason in
  `joint_provenance`.
- `humanart_original_source_oob_v1` pins exactly seven known original-annotation
  defects (stem, annotation ID, joint, raw coordinate, and source dimensions).
  It fails the active dataset audit/build for unexpected, missing, or altered
  visible source-OOB events. The seven are reward-masked with
  `source_coordinate_out_of_bounds`, never clamped or repaired.
- Reconstruction explicitly consumes `keypoints_source` in the raw source
  frame, then applies exact persisted PIL resize/crop. Reward provenance is a
  separate consumer. Direct fixed-stroke final-frame rerender remains
  diagnostic only.
- Primary reconstruction gate: threshold-10 foreground IoU >= 0.63,
  symmetric foreground mean distance <= 2.75 px, p95 <= 3 px. These are
  calibrated from the deterministic 16-per-available-source baseline:
  IoU 0.6380–0.8321, mean distance 0.0917–2.6909 px, p95 1–3 px.

## Verified results

- Source audit PASS: 17,416 active samples; 15,161 available and 2,255
  unavailable (all Danbooru). Diagnostic annotations PASS 21/21; the three
  diagnostic Danbooru stems are all unavailable.
- OOB policy PASS: exactly 7 visible source-OOB joints, exactly two affected
  stems, 0 unexpected, 0 missing reviewed, 0 altered reviewed.
  `painting_humanart_2000000000804` / `2000000007651`: left wrist, left/right
  knee, left/right ankle. `sculpture_humanart_14000000001208` /
  `14000000088574`: left/right ankle.
- New sidecar PASS: `/tmp/pose_targets_v3_authoritative_20260828`, 17,416
  records, SHA-256 `c98f76284179c781f5a14791d66e29dcf5b526168ca79922a40b70af972e444c`.
  It contains 444,235 valid reward joints and 7 source-OOB-masked joints.
- Full reconstruction audit PASS (64 records; 16 per available source):
  `/tmp/pose_target_reconstruction_v3_calibrated_20260828`.
  Mean IoU@10 / mean symmetric distance / max p95: COCO 0.70049 / 0.62002 /
  2.82843; Human-Art painting 0.73069 / 0.33715 / 3.0; real-human 0.71121 /
  0.30531 / 3.0; sculpture 0.75932 / 0.16807 / 2.0.

## Files changed this session

- `pose_controlnet/pose_targets.py`
- `pose_controlnet/control_reconstruction.py`
- `scripts/audit_pose_target_sources.py`
- `scripts/audit_control_reconstruction.py`
- `tests/test_pose_targets.py`
- `docs/POSE_TARGET_SIDECAR.md`
- `docs/CODEX_HANDOFF.md`

## Commands/tests

- PASS: `uv run python -m unittest tests.test_pose_targets tests.test_reference_pose` (27 tests).
- PASS: `python scripts/audit_pose_target_sources.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --authoritative-jsonl data/pose_targets_authoritative_v1.jsonl`.
- PASS: `python scripts/build_pose_target_sidecar.py --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents --authoritative-jsonl data/pose_targets_authoritative_v1.jsonl --output /tmp/pose_targets_v3_authoritative_20260828`.
- PASS: `python scripts/audit_control_reconstruction.py --sidecar /tmp/pose_targets_v3_authoritative_20260828 --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --output-dir /tmp/pose_target_reconstruction_v3_calibrated_20260828 --per-source 16`.
- PASS: `git diff --check` before final handoff update; rerun it before review.

## Exact next recommended action

Review this bounded provenance/reconstruction change. No scientific blocker
remains for this milestone; pose reward remains intentionally unimplemented
and the broader production gates remain separate work.
