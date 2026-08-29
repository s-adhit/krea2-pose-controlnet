# Project handoff

## Current objective and enforced decisions

`train.py` remains unchanged and production training remains flow-matching MSE
only. Gate D is read-only tooling. Gate E is the separately invoked, bounded
Gaussian-heatmap-KL smoke continuation only: `lambda_pose=2e-5`, inclusive
pose timestep window `[0.10, 0.20]`, microbatch size `1`, gradient
accumulation `32`, and target effective batch `32`. Do not launch either tool
from the Codex sandbox.

The immutable Gate-E source parent remains
`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`
with SHA256
`6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`.
The graceful Gate-E checkpoint at
`/lambda/nfs/adhit/krea2-pose/checkpoints/gate-e-parent1500-kl-l2e5-t010-020-mb1-ga32/step_001610.pt`
is the current continuation parent. Desired final global step: `1700`.

## Verified gates and decisions

- Gate A, A.5, B, and C: PASS as previously documented. Gate D remains
  IMPLEMENTED / GH200 RUN REQUIRED.
- Gate E: tooling supports safe dynamic continuation, but it is **not PASS**.
  The real GH200 continuation to step 1700 and its evaluation/inspection are
  still required.
- Gate-E checkpoints now store top-level `gate_e` metadata: pose loss/window,
  critical model/training config, and trainable state names. A later checkpoint
  must match it before it can resume. The already-written legacy step 1610 is
  accepted only after its `metrics.jsonl` proves a single consistent
  `lambda_pose` and timestep window, while its stored full config proves the
  remaining critical config.
- Canonical step 1500 always verifies its pinned SHA (an explicitly supplied
  SHA is an additional check). A later Gate-E checkpoint does not require the
  canonical SHA, but an explicit SHA is enforced if supplied.
- `--target-global-step` is the explicit final-step interface and must exceed
  the loaded checkpoint step. Legacy `--max-steps` means *additional* updates
  from the loaded checkpoint; exactly one of the two is required and each
  continuation is capped at 300 updates.
- New Gate-E runs require a new output directory. Resumes must use exactly the
  parent checkpoint's existing run directory. Published checkpoint names are
  fail-closed and are never overwritten. Full model/optimizer/scheduler,
  epoch/batch position, RNG, and flow-generator restoration still uses the
  existing `train.restore_full_training_state` machinery.
- Optimizer-step metrics now aggregate all accumulation microbatches, including
  active/eligible sample and active-microbatch counts, mean/max active pose
  loss, mean flow/total loss, and timestep min/max/mean. This is observability
  only and does not change RNG draws, loss construction, backward scaling,
  optimizer behavior, VAE/critic behavior, or frozen-boundary checks.

## Files changed this session

- `pose_controlnet/checkpointing.py`
- `pose_controlnet/pose_reward_tools.py`
- `scripts/train_pose_reward_smoke.py`
- `tests/test_pose_reward_tools.py`
- `docs/CODEX_HANDOFF.md`

Existing untracked Gate-B/C/D/E audit files remain user-owned and were not
overwritten. `train.py` was not changed.

## Tests and checks

- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest
  tests.test_pose_reward_tools tests.test_train_mechanics` — 61 CPU tests.
  Covers canonical SHA validation, arbitrary intermediate metadata resume,
  legacy metrics-provenance resume, explicit intermediate SHA, target semantics,
  incompatible configuration rejection, destination/publication safety, and
  all-accumulation diagnostic aggregation/RNG/backward-scaling invariance.
- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest
  tests.test_keypoint_critic tests.test_keypoint_critic_audit
  tests.test_pose_reward_tools` — 41 CPU tests.
- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile
  pose_controlnet/checkpointing.py pose_controlnet/pose_reward_tools.py
  scripts/train_pose_reward_smoke.py tests/test_pose_reward_tools.py`.
- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python
  scripts/train_pose_reward_smoke.py --help`.
- PASS: `git diff --check`.

## Exact next GH200 action (do not run from Codex)

Resume the existing Gate-E run; do not pass the original step-1500 SHA for
this later checkpoint:

```bash
PYTHONPATH=. python scripts/train_pose_reward_smoke.py \
  --parent-checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/gate-e-parent1500-kl-l2e5-t010-020-mb1-ga32/step_001610.pt \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints \
  --run-name gate-e-parent1500-kl-l2e5-t010-020-mb1-ga32 \
  --lambda-pose 2e-5 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --target-global-step 1700 --save-every 50 \
  --microbatch-size 1 --gradient-accumulation-steps 32 --device cuda
```

Inspect the resumed metrics and checkpoint(s), then complete the required
evaluation before considering Gate E PASS.
