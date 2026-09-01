# Project handoff

## Current objective and state

The finishing pose-anneal endpoint is repaired. Its fixed scientific contract
is unchanged: global update 4001 uses `lambda_pose=.04`; update 4500 uses
literal `lambda_pose=0.0`. The final update now legally executes, optimizing
the flow-matching loss alone, and the existing final checkpoint condition
writes `step_004500.pt`.

The only code semantic change is in
`combine_flow_and_pose_loss`: a finite `lambda_pose == 0` is accepted after
the existing active-reference consistency checks and returns the original
`flow_loss` tensor unchanged. Positive values retain the prior
`flow_loss + lambda_pose * pose_loss` calculation. Negative, NaN, and infinite
values fail closed. A zero-active batch still requires `pose_loss is None`.

The ordinary locked production recipe remains `.04`; the constant-lambda
finishing control branch remains `.04` through update 4500. The cooldown
branch and exact-resume identity checks are unchanged and still fail closed.
No training, real evaluation, image generation, network activity, upload,
commit, or push occurred in this maintenance session.

## Exact finishing contract

Both finish branches run updates `4001..4500` inclusive from the fixed step
4000 cooldown parent. The finishing linear schedule is
`lambda_pose(s)=.04*(1-(s-4001)/499)`, so it is exactly `0.0` at update 4500.
The LR remains the existing cosine schedule from `2e-5` to `5e-6`. Final
checkpoint/mirror milestones remain `4100, 4200, 4300, 4400, 4500`.

The production loop now calls a small `checkpoint_due` helper with its
pre-existing condition: checkpoint on save cadence, final `max_steps`, or
controlled stop. This makes the real step-4500 checkpoint boundary directly
CPU-testable without changing its behavior.

## Existing anneal branch recovery (do not run from Codex)

The latest valid local branch checkpoint is expected to be step 4400. Recover
only from persisted checkpoint state; the unsaved in-memory 4499 state cannot
be recovered. This replays updates `4401..4500` deterministically:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py \
  --run-name pose-control-finish-anneal-4000-to4500 \
  --max-steps 4500 --save-every 100 --diagnostics-every 50 \
  --continue-from /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt \
  --continue-from-step 4000 --lr-schedule cosine --lr-start 2e-5 --lr-final 5e-6 \
  --pose-lambda-schedule linear --pose-lambda-final 0 \
  --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-projects \
  --wandb-name pose-control-finish-anneal-4000-to4500 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 100 \
  --resume auto
```

W&B may reject replayed metric rows below an already-recorded remote step; its
failure-isolated mirror must not affect optimizer execution or local checkpoint
creation. Do not clean up remote history as part of recovery.

## Verification this session

PASS: `PYTHONPATH=. python -m unittest tests.test_pose_reward_tools
tests.test_pose_reward_wandb tests.test_production_training -v` — 59 CPU,
no-network tests. New coverage proves `.04` combination behavior, exact-zero
flow-only behavior with active and inactive pose examples, negative/NaN/Inf
rejection, literal zero at step 4500, optimizer update 4500, the live
checkpoint condition, and atomic save/reload of `step_004500.pt`. Existing
production/cooldown/finishing tests prove the unchanged control branch,
ordinary/cooldown update semantics, and fail-closed exact resume.

PASS: `python -m py_compile pose_controlnet/pose_reward_tools.py
pose_controlnet/production_training.py tests/test_pose_reward_tools.py
tests/test_production_training.py`.

Files changed this session: `pose_controlnet/pose_reward_tools.py`,
`pose_controlnet/production_training.py`, `tests/test_pose_reward_tools.py`,
`tests/test_production_training.py`, and this handoff. The pre-existing
untracked `docs/evaluation/pose-control-finish-control-4000-to4500/` remains
untouched.

Next action: review the patch, then run the recovery command above only with
explicit training authorization.
