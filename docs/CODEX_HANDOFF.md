# Project handoff

## Current objective

Implement, but do not launch, the isolated controlled pose-reward exposure continuation. `train.py` remains unchanged and production remains flow-MSE only. Prior Gate-E was technically successful but did not improve external pose: step-1500 Turbo CLIP/PCK(.05/.10/.20) was `.33684298 / .05461 / .18083 / .41262`; Gate-E step-1700 was `.33441689 / .04369 / .14199 / .37257`. Its resumed exposure was only `20 / 2504 = 0.7987%` eligible samples (`18 / 90` optimizer steps): under-exposure, not evidence against Gaussian KL.

## New branch contract

- Parent is only the immutable local step-1500 checkpoint `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`, SHA-256 `6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`. Never continue from the historical Gate-E step-1700 branch.
- Run/remote namespace is exactly `pose-reward-kl-exposure5pct-l2e5-t010-020`. New starts fail if that local directory already exists; resumed checkpoints must be from that same directory and carry format-2 controlled-branch metadata. No checkpoint is overwritten.
- Gaussian heatmap KL, temperature `1.0`, `lambda_pose=2e-5`, final window `[0.10, 0.20]`, microbatch `1`, accumulation `32`, target step `1700`, and saves/mirrors every `50` steps are fixed. Phase-1, evaluator, `make_flow_pair`, and production flow are untouched.
- `--forced-pose-exposure-probability` is required. At `0.05`, only available samples may be selected. Normal timesteps are always drawn first; selected samples receive a uniform final-window timestep; non-forced samples retain normal results. At zero, no extra RNG draws occur. Natural activity excludes forced samples.
- Metadata fail-closes on loss/lambda/window/probability/policy, parent, HF namespace, critical config, and trainable state. It saves cumulative eligible/forced/natural/total counters plus flow-generator state.

## Optional W&B logging

- `wandb>=0.19` was already present in `pyproject.toml`/`uv.lock`; the audited environment has `wandb 0.28.2`. No dependency change was made.
- W&B is disabled unless `--wandb-project` is explicit. JSONL remains the canonical local telemetry, and W&B is only a secondary mirror: W&B failures warn and disable further W&B calls without affecting optimizer work, NFS/local checkpoints, or HF mirroring.
- New enabled checkpoints save `gate_e.wandb_run_id`; enabled resumes use it with W&B `resume="allow"`. If W&B flags are omitted during recovery, local resume remains valid and the ID is preserved. Legacy checkpoints without this optional field remain loadable.
- Suggested project/run: `krea2-pose-controlnet` / `pose-reward-kl-exposure5pct-l2e5-t010-020`. The W&B config is non-secret and includes the immutable parent SHA, Krea-2 Raw identity, branch hyperparameters, cadence/HF target, and sidecar records SHA.

## HF mirror and recovery

Repository: `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`. Remote full checkpoints are exactly:

```text
pose-reward-kl-exposure5pct-l2e5-t010-020/full/step_001550.pt
pose-reward-kl-exposure5pct-l2e5-t010-020/full/step_001600.pt
pose-reward-kl-exposure5pct-l2e5-t010-020/full/step_001650.pt
pose-reward-kl-exposure5pct-l2e5-t010-020/full/step_001700.pt
```

Each is locally saved/validated before HF queueing with a checksum completion marker. Local files are retained; failures are loud after the safe local save and retryable. Final metrics and metadata/config are mirrored too; no token is logged or serialized.

Resume from the newest valid local branch checkpoint by replacing `--parent-checkpoint` below (for example with `.../step_001600.pt`) and removing `--expected-parent-sha256`; retain all other flags. To recover remotely, use `fetch` below then pass its printed path. Never overwrite an existing `step_001700.pt`; evaluate it.

## Exact GH200 command with W&B — do not run from Codex

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
  --run-name pose-reward-kl-exposure5pct-l2e5-t010-020 \
  --lambda-pose 2e-5 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --forced-pose-exposure-probability 0.05 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --hf-subdir pose-reward-kl-exposure5pct-l2e5-t010-020 \
  --hf-mirror-every-steps 50 --target-global-step 1700 --save-every 50 \
  --microbatch-size 1 --gradient-accumulation-steps 32 --device cuda \
  --wandb-project krea2-pose-controlnet \
  --wandb-run-name pose-reward-kl-exposure5pct-l2e5-t010-020
```

Fallback with W&B disabled: use the exact command above with the final two `--wandb-*` lines omitted.

Verify/list the mirror:

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py list \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-kl-exposure5pct-l2e5-t010-020
```

Retry an already-saved checkpoint (and verify its marker):

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py mirror \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-kl-exposure5pct-l2e5-t010-020 \
  --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-reward-kl-exposure5pct-l2e5-t010-020/step_001650.pt
```

Fetch a marker-validated remote checkpoint for recovery:

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py fetch \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-kl-exposure5pct-l2e5-t010-020 --step 1650 \
  --download-dir /lambda/nfs/adhit/krea2-pose/recovery/pose-reward-kl-exposure5pct-l2e5-t010-020
```

## Checks this session

- PASS: `PYTHONPATH=. python -m unittest tests.test_pose_reward_wandb tests.test_pose_reward_tools tests.test_timestep_exposure tests.test_gate_e tests.test_wandb_logging` — 47 CPU/no-network tests, including W&B failure isolation and run-ID resume.
- PASS: `PYTHONPATH=. python -m py_compile scripts/train_pose_reward_smoke.py pose_controlnet/wandb_logging.py tests/test_pose_reward_wandb.py`.
- PASS: `PYTHONPATH=. python scripts/train_pose_reward_smoke.py --help`.
- PASS: `git diff --check`.

Files changed this session: `scripts/train_pose_reward_smoke.py`, `pose_controlnet/wandb_logging.py`, `tests/test_pose_reward_wandb.py`, and this handoff. No training, W&B login, or network call was made.

## After step 1700

Run the fixed Turbo evaluator on only these four checkpoints; compare CLIP and authoritative PCK with the preserved step-1500 numerical baseline. Do not alter historical Gate-E/Phase-1 artifacts.
