# Project handoff

## Current objective and decisions

This pass re-baselines the abandoned RTMPose/MMPose critic experiment. It was
abandoned before reward training: no RTMPose-based reward branch was trained.
The reason is integration/dependency complexity and the decision to standardize
the next Phase-2 direction on DWPose only. No DWPose critic is implemented in
this repository yet.

Phase-1 provenance remains authoritative and unchanged. In particular,
`data/pose_targets_authoritative_v1.jsonl` remains the source for active
COCO/Human-Art targets; sidecar v3 retains source-coordinate keypoints,
`reward_joint_valid` masks, and the exact seven Human-Art visible source-OOB
anomalies, which are masked rather than repaired. Danbooru remains explicitly
flow-only (`pose_reward_available=false`). Control reconstruction, paired
preprocessing, dataset indexing, and external Keypoint R-CNN PCK evaluation
remain unchanged.

## Verified Phase-1 state

- Source audit: 17,416 active samples; 15,161 pose-reward available and 2,255
  unavailable (all Danbooru). Diagnostic annotations passed 21/21.
- OOB contract: exactly 7 visible source-OOB joints in two Human-Art stems;
  no unexpected, missing, or altered reviewed events.
- Sidecar build: 17,416 records, SHA-256
  `c98f76284179c781f5a14791d66e29dcf5b526168ca79922a40b70af972e444c`, with
  444,235 valid reward joints and 7 source-OOB-masked joints.
- Reconstruction audit (64 records): all calibrated gates passed.

## DWPose baseline

No tracked DWPose runtime, preprocessing utility, model wrapper, or dependency
exists. The only tracked DWPose mention is a Phase-1 test asserting that the
obsolete `historical_dwpose_jsonl` pseudo-label fallback is absent. Therefore
the current repository exposes no DWPose raw heatmaps/logits and has no
established input-gradient path; differentiability, raw-output availability,
and detector/NMS/argmax boundaries must be audited before it can be used as a
training reward.

`.venv/` and the harmless historical `.venv-rtmpose/` remain ignored.

## Files changed in this cleanup pass

- Deleted: `pose_controlnet/pose_critic.py`, `scripts/audit_pose_critic.py`,
  `scripts/audit_pose_critic_vae.py`, `tests/test_pose_critic.py`, and
  `docs/POSE_CRITIC_AUDIT.md`.
- Modified: `docs/CODEX_HANDOFF.md`.

## Validation and blockers

- PASS: `PYTHONPATH=. python -m unittest tests.test_pose_targets
  tests.test_reference_pose tests.test_post500_evaluation
  tests.test_post1500_evaluation` (45 tests).
- PASS: `git diff --check`.
- No Python file was modified in this cleanup pass (the only Python changes are
  deletions), so `py_compile` is not applicable. No GPU job was run.
- Remaining tracked RTMPose/MMPose search hits are intentional historical
  mentions in this handoff and `.venv-rtmpose/` in `.gitignore`; the latter is
  retained as a harmless historical ignore. A binary evaluation contact sheet
  produces an incidental byte-level `git grep` match but contains no tracked
  implementation or dependency reference.
- Blocker: DWPose suitability has not been audited. Do not add dependencies or
  implement a critic until the audit establishes an approved backend, direct
  raw-output access, and an end-to-end gradient path from the frozen pose model
  to its RGB input without detector/NMS/argmax boundaries in the reward path.

## Exact next recommended action

Perform a DWPose-only, no-training audit: inventory the approved DWPose model
and backend, trace preprocessing and person detection/postprocessing, then run
a minimal frozen-model RGB-input autograd probe that records raw tensor names,
shapes, and finite nonzero input-gradient norm. Preserve Phase-1 sidecar and
external Keypoint R-CNN PCK behavior unchanged.
