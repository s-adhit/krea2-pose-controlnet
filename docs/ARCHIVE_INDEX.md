# Archive index

This index separates the canonical production surface from preserved historical
work. Archive status does not change a run's checkpoint, data, or scientific
provenance.

## Canonical surfaces

- Training: `scripts/train_production.py` and `pose_controlnet/production_training.py`.
  The locked recipe is flow-matching MSE plus normalized-coordinate
  pose-consistency Huber (`lambda_pose=0.04` for the main production/control
  branch), including its recorded timestep exposure and resume behavior.
- Inference: `inference.py`, using `pose_controlnet/turbo_runtime.py`.
- Shared geometry: `pose_controlnet/resolution_policy.py`.
- Current candidates: `parent-4000` (balanced) and `finish-control-a4300`
  (pose specialist). `finish-anneal-b4200` and the anneal branch are historical
  only and must not appear in current candidate lists.
- Evaluation terminology: diagnostic is the development/selection benchmark;
  validation is held out from training but used for inference benchmarking, not
  an untouched final test set.

## Historical code and evidence

- `scripts/train_pose_reward_smoke.py`, overfit/capacity utilities, critic
  audits, and Turbo benchmark utilities remain historical experiment tooling.
  Reusable production pose-consistency math now lives in
  `pose_controlnet/pose_consistency.py`.
- `pose_controlnet/turbo_evaluation.py` retains historical evaluation contracts
  and re-exports the locked sampling/runtime helpers from
  `pose_controlnet/turbo_runtime.py` for provenance-compatible callers.
- Obsolete development prompt-transfer evidence is retained under
  `docs/archive/inference_eval/`. Current evidence remains in
  `docs/inference_smoke/`, `docs/inference_eval/a4300-krea-native-matched/`,
  and `docs/inference_eval/val_pose_candidates/`.
- `docs/evaluation/`, `configs/overfit_capacity/`, and historical test files
  remain preserved in place because their paths are referenced by recorded
  experiment metadata and tests. They are archival material, not canonical
  launch surfaces.

## Redistribution review

Committed Human-Art-derived imagery is preserved without a rights decision.
Review these exact repository subtrees before redistribution:

- `docs/inference_eval/a4300-krea-native-matched/`
- `docs/evaluation/gate-e-kl-l2e5-t010-020/`
- `docs/evaluation/pose-learning-100/`
- `docs/evaluation/pose-reward-coord-exposure10pct-l1e5-t010-020/`
- `docs/evaluation/pose-reward-kl-exposure10pct-l1e5-t010-020/`
- `docs/evaluation/pose-reward-kl-exposure10pct-l1e5-t020-030/`
- `docs/evaluation/pose-reward-kl-exposure5pct-l2e5-t010-020/`
- `docs/evaluation/pose-reward-kl-exposure5pct-l2e5-t010-020-WRONG-OLD-GATE-E/`
- `docs/evaluation/pose-reward-kl-exposure5pct-l2e5-t010-020-overnight2500/`
