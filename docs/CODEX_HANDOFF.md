# Project handoff

## Current objective

Gate-E pose-reward smoke trainer runtime repair is complete; do not launch training from Codex. `train.py` is untouched; production remains flow-MSE only.

## Runtime repair completed

Root cause: `scripts/train_pose_reward_smoke.py::main` initially stored the checkpoint `Path` in `parent`, then rebound `parent` to the loaded checkpoint dictionary. The later new-run directory setup evaluated `parent.parent`, causing `AttributeError: 'dict' object has no attribute 'parent'` after W&B/model setup.

Fix: non-GPU setup now returns a `GateERunSetup` with explicitly separate `parent_path` and `parent_state`. Path comparison/directory creation use only `parent_path`; configuration, canonical provenance, controlled-resume validation, W&B run-id recovery, model loading, and full state restoration use only `parent_state`. SHA validation, controlled-branch detection, fail-closed destination behavior, W&B/HF behavior, checkpoint schema, optimizer, RNG, timestep sampling, forced exposure, and lambda remain unchanged.

Focused CPU/no-network regression: `test_non_gpu_setup_keeps_checkpoint_path_and_state_separate_for_new_and_resume` executes both canonical new-start and intermediate controlled-resume setup without model loading or W&B/HF calls.

Artifact inspection: no artifacts were deleted. In the current Codex filesystem, the intended run directory, `step_001525.pt`, and `experiment_metadata.json` are absent under `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-reward-kl-exposure5pct-l1e5-t010-020`. Recheck from the actual GH200 operator shell before rerunning, since the failed invocation may have created metadata outside this view.

## Next controlled run — do not run from Codex

Immutable parent:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt
sha256: 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0
```

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_pose_reward_smoke.py \
  --parent-checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt \
  --expected-parent-sha256 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0 \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints \
  --run-name pose-reward-kl-exposure5pct-l1e5-t010-020 \
  --lambda-pose 1e-5 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --forced-pose-exposure-probability 0.05 \
  --target-global-step 1650 --save-every 25 --microbatch-size 1 --gradient-accumulation-steps 32 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --hf-subdir pose-reward-kl-exposure5pct-l1e5-t010-020 --hf-mirror-every-steps 25 --device cuda \
  --wandb-entity adhit-projects --wandb-project Krea-2-PoseControl-Lora \
  --wandb-run-name pose-reward-kl-exposure5pct-l1e5-t010-020
```

Start an interactive tmux shell, then paste that command:

```bash
tmux new-session -s pose-reward-kl-exposure5pct-l1e5-t010-020 'cd /home/ubuntu/krea2-pose-controlnet && exec bash'
```

Expected local/HF steps: `1525 1550 1575 1600 1625 1650`. Check remote completion markers with:

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py list \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-kl-exposure5pct-l1e5-t010-020
```

Resume: set `--parent-checkpoint .../pose-reward-kl-exposure5pct-l1e5-t010-020/step_00XXXX.pt`, omit `--expected-parent-sha256`, and preserve all other experiment flags.

## Checks this session

- PASS: `PYTHONPATH=. python -m unittest tests.test_pose_reward_tools tests.test_pose_reward_wandb tests.test_train_mechanics` (70 tests: Gate-E new/resume, W&B, and training-resume mechanics).
- PASS: `PYTHONPATH=. python -m py_compile scripts/train_pose_reward_smoke.py tests/test_pose_reward_tools.py tests/test_pose_reward_wandb.py`.
- PASS: `PYTHONPATH=. python scripts/train_pose_reward_smoke.py --help`.
- PASS: `git diff --check`.

Changed this session: `scripts/train_pose_reward_smoke.py`, `tests/test_pose_reward_tools.py`, `docs/CODEX_HANDOFF.md`. No training, evaluation, network call, commit, or push.
