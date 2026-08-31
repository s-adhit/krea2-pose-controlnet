# Project handoff

## Current objective and status

The full-dataset production launcher is implemented at
`scripts/train_production.py`. It is separate from the bounded Gate-F
`train.py` entry point and never enables or uses `--allow-extended-training`.

Observed real GH200 smoke: steps 1–5 completed and atomically wrote
`step_000005.pt`; `--resume auto` restored it and completed steps 6–7, writing
`step_000007.pt`. This verified basic model/optimizer/local-checkpoint/resume
operation but exposed restarted pose cumulative counters. Production checkpoints
now persist/restore `eligible_samples_seen`, `naturally_active_samples`,
`forced_samples`, and `total_active_samples`; legacy checkpoints can recover
their exact checkpoint-step counters from local `metrics.jsonl`. This does not
alter activation or timestep sampling.

No real training, generation, evaluation, long GPU benchmark, commit, or push
occurred in the observability/counter-fix session. The full 16,503-sample 768
cache and pose sidecar were not opened or modified by its CPU/no-network tests.

## Locked production recipe

- Dataset snapshot: `/lambda/nfs/adhit/krea2-pose/posebridge_hf`
- Authoritative manifest: `data/manifests/train.jsonl`
- Cache: `/lambda/nfs/adhit/krea2-pose/posebridge_latents_768`
- Text cache: `/lambda/nfs/adhit/krea2-pose/text_conditioning`
- Pose sidecar: `/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768`
- Raw checkpoint: `/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`
- Frozen Krea-2 Raw + ControlInputLayer + rank/alpha 64 LoRA, existing 224
  target topology.
- Objective: existing flow MSE plus `normalized_coordinate_huber`,
  `lambda_pose=0.04`, natural pose window `[0.10, 0.20]`, forced exposure 0.
- AdamW `lr=1e-4`, betas `(0.9, 0.99)`, weight decay 0, existing max-grad
  norm `1.0`, 200 optimizer-step warmup.
- Microbatch 1, accumulation 32, effective batch 32; seed 42; BF16.
- Selected GH200 runtime: loader workers 4, persistent workers, pinned memory,
  prefetch factor 4; gradient checkpointing/compile/fused AdamW all disabled.

The launcher exposes the runtime controls but fail-closes overrides: this is a
locked production recipe, not a new experiment interface. `--max-steps` and
`--run-name` are required; 3000 steps are accepted without the Gate-F 100-step
limit. Checkpoints default to every 250 optimizer steps.

Benchmark conclusion: baseline was about 19.646 sec/optimizer step; loader4
about 15.786 sec/step. Microbatch 2 was slower, compile was neutral/slightly
slower, and flow-only had no meaningful advantage. Therefore loader4 is the
production setting.

## Preflight, checkpoint, and resume contract

Before CUDA/model/optimizer construction, `verify_full_768_cache` validates
the complete cache and authoritative pose sidecar. It fails closed on cache
completion, count, immutable manifest identity/order, 768 geometry policy,
cache-contract, sidecar membership/geometry, and sidecar-record identity.

`run_metadata.json` and every atomic `step_*.pt` record the scientific recipe,
cache contract SHA, sidecar records SHA, manifest record/order hashes, raw
checkpoint/text-cache hashes, optimizer/scheduler/loader configs, code revision
when available, current/max step, and run name. Checkpoints include trainable
ControlInput/LoRA state, optimizer, scheduler, global step, epoch/batch/sample
position, accumulation position, Python/NumPy/Torch CPU/CUDA RNG, and the
flow/timestep generator state. Resume restores all state and rejects changed
recipe, artifact identities, position metadata, missing generator state, or a
checkpoint beyond the requested maximum. `--resume auto` is local-only and
has no network dependency.

## Optional production observability

W&B is disabled by default (`--no-wandb`) and, when disabled, does not import
or initialize W&B. `--wandb` enables a failure-isolated mirror;
`--wandb-project` defaults to `Krea-2-PoseControl-Lora` and `--wandb-name`
defaults to the run name. It mirrors the already-collected JSONL step metrics,
including losses, learning rate, gradient norm, timing, pose/cumulative
counters, and timestep diagnostics, without new CUDA synchronization. The W&B
run ID is checkpointed: a local resume with W&B enabled passes that ID with
`resume="allow"` to continue the same remote run. Init/log/finish failures only
disable W&B for that process; local JSONL/checkpoints continue.

HF mirroring is disabled by default (`--hf-repo-id ''`,
`--hf-mirror-every-steps 0`). With both options supplied, only atomically
published and deserialize-validated `step_*.pt` files are queued. The existing
private-repo helper uploads the full checkpoint then its checksum completion
marker; the checkpoint contains the production provenance. Failures are retried
and reported while the local checkpoint remains authoritative. Temp/incomplete
files are rejected. Local saves every 250 plus HF every 500 gives exact 3000
milestones: `500, 1000, 1500, 2000, 2500, 3000`.

## Verification completed this session

PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_production_training tests.test_production_throughput_benchmark -v
PYTHONPATH=. python -m py_compile pose_controlnet/production_training.py scripts/train_production.py tests/test_production_training.py
PYTHONPATH=. python -m unittest tests.test_production_training tests.test_train_mechanics tests.test_pose_reward_wandb -v
```

The focused CPU/no-network tests cover the locked CLI/defaults, batch 32,
200-step warmup, exact LR/pose recipe, loader4 defaults, disabled runtime
alternatives, scientific resume identity, default-off W&B/HF behavior, W&B
checkpoint run-ID continuity, exact 500-step HF milestones, nonfatal HF
failure/local authority, temp-checkpoint rejection, and cumulative-counter
continuity.

## Exact 3000-step operator commands (do not run from Codex)

```bash
cd /home/ubuntu/krea2-pose-controlnet
tmux new-session -d -s pose-production-3000 "cd /home/ubuntu/krea2-pose-controlnet && mkdir -p /lambda/nfs/adhit/krea2-pose/production-logs && set -o pipefail && PYTHONPATH=. python scripts/train_production.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --train-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents_768 --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --pose-sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3_768 --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints --run-name pose-control-production-3000 --max-steps 3000 --save-every 250 --diagnostics-every 50 --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-name pose-control-production-3000 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 500 2>&1 | tee /lambda/nfs/adhit/krea2-pose/production-logs/pose-control-production-3000.log"

tmux attach -t pose-production-3000
tail -F /lambda/nfs/adhit/krea2-pose/production-logs/pose-control-production-3000.log
watch -n 10 'ls -lh /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000; tail -n 2 /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000/metrics.jsonl'
```

Resume after an interruption:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --train-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents_768 --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --pose-sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3_768 --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints --run-name pose-control-production-3000 --max-steps 3000 --save-every 250 --diagnostics-every 50 --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-name pose-control-production-3000 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 500 --resume auto
```

## Files changed this session

- `pose_controlnet/production_training.py`
- `scripts/train_production.py`
- `tests/test_production_training.py`
- `docs/CODEX_HANDOFF.md`

## Next action

Review the launcher and test patch, then perform the outstanding real GH200
preflight/stability and service gates before asking for authorization to launch
the 3000-step run.
