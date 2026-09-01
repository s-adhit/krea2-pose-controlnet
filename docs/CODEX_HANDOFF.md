# Project handoff

## Current objective and canonical surfaces

Final production-dependency cleanup and README rewrite completed locally; no
training, GPU inference/evaluation, network access, commit, or push occurred.

- Canonical training entrypoint: `scripts/train_production.py`, backed by
  `pose_controlnet/production_training.py`.
- Canonical user-facing inference CLI/API: `inference.py`.
- Canonical production objective: flow-matching MSE plus explicit
  normalized-coordinate pose-consistency Huber. The main production/control
  branch is `lambda_pose=0.04`, with the existing controlled timestep exposure
  behavior and resumable checkpoint semantics preserved.
- Exact reusable implementation: `pose_controlnet/pose_consistency.py`,
  function `production_pose_consistency_loss` (with the accumulation
  diagnostics and cumulative exposure counters beside it).  It uses the
  small `PoseConsistencyRuntimeConfig` protocol, not `train.TrainConfig`.
- Neutral production dependencies are `pose_controlnet/pose_critic.py`,
  `pose_controlnet/pose_loss.py`, and `pose_controlnet/training_runtime.py`.
  `pose_controlnet/keypoint_critic.py` and `pose_controlnet/pose_reward_tools.py`
  retain historical compatibility re-exports only.
- Shared 768 geometry lives in `pose_controlnet/resolution_policy.py`.
- Locked Turbo sampling/runtime helpers live in `pose_controlnet/turbo_runtime.py`.
  `pose_controlnet/turbo_evaluation.py` remains a historical evaluation layer
  and re-exports those helpers for compatibility.

Current checkpoint status: `parent-4000` is the balanced candidate and
`finish-control-a4300` is the pose-specialist candidate. The entire anneal
branch, including B4200, is historical only and is absent from the current
inference candidate list.

Terminology: diagnostic is the development/selection benchmark. Validation is
held out from training but is used for inference benchmarking; it is not an
untouched final test set.

## Audit changes

- `production_training.py` no longer imports `scripts/train_pose_reward_smoke.py`.
  The historical smoke script delegates to the reusable library implementation.
- `production_training.py` and `pose_consistency.py` no longer import
  `train.py`, `keypoint_critic`, `pose_reward_tools`, or critic-audit helpers.
  Production uses the neutral runtime, fixed-box critic, and loss modules.
- Production-facing inference no longer imports bucket policy or locked runtime
  helpers from experiment modules.
- Removed empty/dead `scripts/prefetch_models.py` and `run_forever.sh`, plus
  invalid placeholder `requirements/local-x86-cuda.txt`.
- Moved obsolete prompt-transfer development evidence to
  `docs/archive/inference_eval/`; current parent/A4300 evidence remains under
  `docs/inference_smoke/`, `docs/inference_eval/a4300-krea-native-matched/`,
  and `docs/inference_eval/val_pose_candidates/`.
- `docs/ARCHIVE_INDEX.md` identifies canonical surfaces, historical material,
  and the Human-Art redistribution-review paths. No Human-Art imagery was
  removed and no rights/licensing decision was made.

## Verification

PASS (CPU, no network):

```bash
PYTHONPATH=. python -m py_compile inference.py \
  pose_controlnet/production_training.py pose_controlnet/pose_consistency.py \
  pose_controlnet/resolution_policy.py pose_controlnet/turbo_runtime.py \
  pose_controlnet/turbo_evaluation.py pose_controlnet/overfit_capacity.py \
  scripts/train_production.py scripts/train_pose_reward_smoke.py \
  scripts/benchmark_production_trainer.py scripts/train_overfit_capacity.py

PYTHONPATH=. python -m unittest tests.test_inference \
  tests.test_production_training tests.test_pose_reward_tools \
  tests.test_pose_reward_wandb tests.test_turbo_evaluation \
  tests.test_capacity_experiment_axes tests.test_production_milestone_evaluation -v
```

94 tests passed. Run `git diff --check` after reviewing the final working tree
before staging. No checkpoint, manifest, or source dataset was changed.

Final dependency-cleanup verification:

```bash
PYTHONPATH=. python -m py_compile pose_controlnet/pose_consistency.py \
  pose_controlnet/pose_critic.py pose_controlnet/pose_loss.py \
  pose_controlnet/training_runtime.py pose_controlnet/keypoint_critic.py \
  pose_controlnet/production_training.py scripts/train_production.py

PYTHONPATH=. python -m unittest tests.test_production_training \
  tests.test_pose_reward_tools tests.test_keypoint_critic tests.test_inference \
  tests.test_turbo_evaluation tests.test_train_mechanics -v
# PASS: 141 tests

PYTHONPATH=. python -m unittest tests.test_control_diagnostics \
  tests.test_turbo_evaluation tests.test_inference tests.test_production_training \
  tests.test_pose_reward_tools -v
# PASS: 87 tests
```

The canonical import direction is now:

```text
scripts/train_production.py -> pose_controlnet/production_training.py
  -> pose_controlnet/{pose_consistency, pose_critic, pose_loss, training_runtime}.py
```

Static import audit found no backwards import from that path to `train.py`,
the historical pose smoke script, `keypoint_critic`, `pose_reward_tools`, or
`keypoint_critic_audit`. `inference.py` contains exactly the current checkpoint
candidates `parent-4000` and `finish-control-a4300`; anneal/B candidates remain
historical only.

## Next action

Review and stage the dependency cleanup plus the rewritten top-level README.
The README documents the current Raw-to-Turbo pose-control workflow,
parent-4000/A4300 candidate status, prompt guidance, and commented-only
showcase placeholders; no assets were created. A redistribution/legal decision
for the Human-Art-derived committed imagery remains explicitly pending; do not
delete or history-rewrite it without approval.
