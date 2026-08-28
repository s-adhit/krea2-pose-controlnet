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

Run the Phase-2 audit-only critic gates in the verified GH200 shell. The
provenance/reconstruction milestone is complete and must remain unchanged.

## Phase 2 audit-only implementation status

- Added `pose_controlnet/pose_critic.py`, which is deliberately not imported
  by `train.py`. It freezes/evals a critic and calls its raw top-down
  `extract_feat`/`head.forward` path directly: no detector, NMS, inferencer,
  argmax, Keypoint R-CNN, DWPose, or control raster input.
- Candidate provenance: official MMPose v1.3.2 RTMPose-M COCO-WholeBody
  `rtmpose-m_8xb64-270e_coco-wholebody-256x192`; config `(W,H)=(192,256)`,
  split ratio `2`, raw SimCC vectors `(384,512)`, Gaussian sigma `(4.9,5.66)`.
  Only output indices 0--16 are used; renderer neck/hands/face/foot-extra are
  excluded.
- Fixed crop uses sidecar-v3 `bbox_training_xywh`, MMPose validation padding
  1.25/aspect correction, differentiable bilinear `grid_sample`, and jointly
  masks provenance-invalid plus SimCC-OOB joints. Candidate losses are
  beta-softmax expectation Huber and `official_simcc_kl`.
- Phase-2 audit correction complete: `CriticSpec` now carries official
  `beta=10`/`label_beta=10`; differentiable expectations, entropy, and
  confidence use beta-softmax probabilities; raw SimCC Gaussian labels remain
  non-normalized before label softmax; and `official_simcc_kl` matches the
  verified KLDiscretLoss formula while averaging only valid joints. Detached
  raw argmax decoding is metric-only. The real-image gradient check uses only
  `official_simcc_kl` and fails if any frozen critic parameter has a gradient.
- Added `scripts/audit_pose_critic.py` for deterministic real RGB 16/source
  metrics, contact sheet, beta-softmax confidence/entropy, separate soft
  expectation and detached raw-argmax PCK/error metrics, both losses, and
  image-gradient checks. It writes only to a supplied external directory.
  Added CPU toy tests and `docs/POSE_CRITIC_AUDIT.md`.

## Current Phase 2 blockers

- Sandbox facts: Python 3.10.12, torch 2.7.0/CUDA 12.8, torchvision 0.22.0,
  diffusers 0.40.0; mmengine/mmcv/mmpose absent; no CUDA visible. This does
  not override host-verified GH200 facts.
- No RTMPose assets are cached. `uv add mmengine==0.10.7 mmcv-lite==2.1.0
  mmpose==1.3.2` and direct GitHub clone both failed before mutation due DNS.
  No SHA was obtained or fabricated. On the networked GH200 host use the
  documented pure-Python official stack, also `mmdet==3.3.0`, stage config and
  weight, record `sha256sum`, then run real-image gate before VAE/x0-hat.

## Phase 2 commands/tests

- PASS: `PYTHONPATH=. python -m unittest tests.test_pose_critic` (9 tests).
- PASS: `python -m py_compile pose_controlnet/pose_critic.py scripts/audit_pose_critic.py tests/test_pose_critic.py`.
- PASS: `git diff --check` after the audit correction.
- Current changed files: `pose_controlnet/pose_critic.py`,
  `scripts/audit_pose_critic.py`, `tests/test_pose_critic.py`,
  `docs/POSE_CRITIC_AUDIT.md`, and this handoff. Untracked `.venv-rtmpose/`
  pre-existed and was not modified.
