# Project handoff

## Current objective

The isolated Gate-E smoke trainer now supports a selectable differentiable pose reward. `train.py` remains untouched and production training remains flow-matching MSE only. Do not launch training or evaluation from Codex.

## Current result and decision

Gaussian KL at 10% exposure, `lambda_pose=1e-5`, window `[0.20,0.30]` did **not** justify overnight: parent PCK@.05/.10/.20 was `.05461/.18083/.41262`; best probe PCK@.10 stayed below it; step 1600 was roughly `.04976/.15534/.38350`. Do **not** increase exposure further.

## Implemented coordinate reward

`--pose-loss` selects compatible-default `gaussian_heatmap_kl` or `normalized_coordinate_huber`. The latter reuses the fixed-box critic: spatial softmax (`T=1`) → expected heatmap coordinates → authoritative ROI cell-center mapping → both prediction/target normalized as `(x-x0)/max(x1-x0,eps)`, `(y-y0)/max(y1-y0,eps)` → Smooth-L1 (`delta=1`) averaged only over `reward_joint_valid`. No argmax, detector, matching, NMS, or PCK is on the backward path. Invalid/OOB joints are zero; Danbooru remains unavailable. The selected value is in checkpoint/experiment metadata, W&B config, fail-closed resume validation, and dynamic Turbo provenance.

## Prepared next experiment — do not run from Codex

Run name / HF namespace / W&B run name: `pose-reward-coord-exposure10pct-l1e5-t010-020`

Immutable parent:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt
sha256: 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0
```

Contract: normalized-coordinate Huber; `lambda_pose=1e-5`; forced exposure `0.10`; pose window `[0.10, 0.20]`; target step `1600`; microbatch `1`; gradient accumulation `32`; save/mirror every `25` steps. Expected checkpoints: `1525 1550 1575 1600`.

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
  --run-name pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --pose-loss normalized_coordinate_huber --lambda-pose 1e-5 \
  --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --forced-pose-exposure-probability 0.10 \
  --target-global-step 1600 --save-every 25 --microbatch-size 1 --gradient-accumulation-steps 32 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --hf-subdir pose-reward-coord-exposure10pct-l1e5-t010-020 --hf-mirror-every-steps 25 --device cuda \
  --wandb-entity adhit-projects --wandb-project Krea-2-PoseControl-Lora \
  --wandb-run-name pose-reward-coord-exposure10pct-l1e5-t010-020
```

Start tmux, then paste the command above:

```bash
tmux new-session -s pose-reward-coord-exposure10pct-l1e5-t010-020 'cd /home/ubuntu/krea2-pose-controlnet && exec bash'
```

Verify the HF mirrors:

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py list \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-coord-exposure10pct-l1e5-t010-020
```

After all four checkpoints exist, run this dynamic Turbo evaluation from the GH200 operator shell:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/turbo_benchmark.py experiment \
  --checkpoint-root /lambda/nfs/adhit/krea2-pose/checkpoints/pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --steps 1525 1550 1575 1600 \
  --output-root docs/evaluation/pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --experiment-name pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --checkpoint-label-template 'Coordinate Huber 1e-5 10pct t010-020 {step}' \
  --baseline-output-root docs/evaluation/turbo-8step-cfg0-lr5e5 \
  --baseline-step 1500 --baseline-label 'LR-only 1500 @ 5e-5' \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints
```

Expected artifacts: `checkpoint_preflight.json`, `turbo_spec.json`, `generation_results.json`, `pck_clip_results.json`, `evaluation_summary.json`, `checkpoint_ranking.json`, the named selection grid/contact sheet, and per-stem `fixed_pose/` files under `docs/evaluation/pose-reward-coord-exposure10pct-l1e5-t010-020/`.

## Checks completed this session

- PASS: `PYTHONPATH=. python -m unittest tests.test_pose_reward_tools tests.test_keypoint_critic tests.test_keypoint_critic_audit tests.test_pose_reward_wandb tests.test_train_mechanics tests.test_turbo_evaluation tests.test_turbo_lr5e5_evaluation tests.test_turbo_timestep_evaluation` (123 CPU/no-network tests).
- PASS: `PYTHONPATH=. python -m py_compile scripts/train_pose_reward_smoke.py pose_controlnet/keypoint_critic.py tests/test_pose_reward_tools.py tests/test_keypoint_critic.py tests/test_pose_reward_wandb.py tests/test_turbo_evaluation.py`.
- PASS: `PYTHONPATH=. python scripts/train_pose_reward_smoke.py --help`.
- PASS: `PYTHONPATH=. python scripts/turbo_benchmark.py experiment --help`.

Changed this session: `pose_controlnet/keypoint_critic.py`, `scripts/train_pose_reward_smoke.py`, focused reward/W&B/Turbo tests, and this handoff. No training, evaluation, network call, commit, or push occurred.
