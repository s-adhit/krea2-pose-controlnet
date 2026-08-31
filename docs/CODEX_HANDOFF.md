# Project handoff

## Current objective and status

The full-dataset production launcher is implemented at
`scripts/train_production.py`. It is separate from the bounded Gate-F
`train.py` entry point and never enables or uses `--allow-extended-training`.

Observed real 10-step production-service smoke atomically saved local
`step_000005.pt` and `step_000010.pt`, and queued the configured step-10 HF
mirror, but no remote checkpoint/marker appeared. Root cause: the mirror worker
was daemonized and `stop()` placed its sentinel then joined for only 30 seconds;
a roughly 2.5-GB final upload could still be in progress when normal process
exit killed the worker.

`HFTrainingCheckpointMirror.stop(drain=True, timeout=None)` now stops new
submissions, appends its sentinel after all accepted FIFO work, and waits until
every queued upload has reached its existing terminal success/failure result.
The worker calls `Queue.task_done()` for every request. Production calls this
explicit draining mode in its `finally` block. The legacy `stop()` form remains
bounded at 30 seconds for older trainers and returns `False` plus a visible
shutdown failure record if it times out. The production drain has no hidden
30-second cutoff. Upload ordering remains full checkpoint followed by its
`.complete.json` marker; local checkpoints remain authoritative after failure;
invalid/temp files remain rejected.

No training, generation, evaluation, long GPU job, real HF upload, commit, or
push occurred in this shutdown-fix session. The full 16,503-sample 768 cache
and pose sidecar were not opened or modified.

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
defaults to the run name. `--wandb-entity` defaults to `None`; when supplied,
it is passed explicitly to `wandb.init(entity=...)` and recorded in the local
run/checkpoint metadata. It mirrors the already-collected JSONL step metrics,
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
files are rejected. At normal production exit, the accepted queue is drained
without a fixed join timeout before the process returns. Local saves every 250
plus HF every 500 gives exact 3000 milestones: `500, 1000, 1500, 2000, 2500,
3000`.

## Verification completed this session

PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_production_training.ProductionTrainingTests.test_wandb_cli_enablement_and_local_checkpoint_resume_identity tests.test_production_training.ProductionTrainingTests.test_cli_defaults_are_the_locked_loader4_recipe -v
PYTHONPATH=. python -m py_compile pose_controlnet/production_training.py scripts/train_production.py tests/test_production_training.py
```

Shutdown-fix verification (CPU/mock/no-network), all PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_train_mechanics.TrainMechanicsTest.test_hf_mirror_queues_exact_paths_while_upload_is_in_flight tests.test_train_mechanics.TrainMechanicsTest.test_hf_mirror_draining_stop_waits_for_slow_final_upload_beyond_legacy_timeout tests.test_train_mechanics.TrainMechanicsTest.test_hf_mirror_draining_stop_preserves_multiple_upload_and_marker_order tests.test_train_mechanics.TrainMechanicsTest.test_hf_mirror_draining_stop_reports_failed_upload_and_completes tests.test_train_mechanics.TrainMechanicsTest.test_hf_mirror_draining_stop_with_no_work_is_prompt tests.test_production_training.ProductionTrainingTests.test_production_trainer_uses_draining_hf_mirror_shutdown -v
PYTHONPATH=. python -m unittest tests.test_train_mechanics tests.test_production_training -v
PYTHONPATH=. python -m py_compile pose_controlnet/checkpointing.py pose_controlnet/production_training.py tests/test_train_mechanics.py tests/test_production_training.py scripts/mirror_checkpoint.py
git diff --check
```

The 66-test combined suite covers queued-work draining, a blocked final upload,
multiple FIFO checkpoint/marker pairs, terminal upload failure with intact local
checkpoint, prompt empty shutdown, existing checkpointing behavior, production
draining invocation, temp-file rejection, and local-authority behavior.

## Direct operator recovery check (do not run from Codex)

Mirror the existing valid step-10 checkpoint synchronously; this helper waits
for the full upload, then the completion marker, then verifies the result:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/mirror_checkpoint.py mirror --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-control-service-smoke-10 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-service-smoke-10/step_000010.pt
```

Verify both expected remote files and the marker/checksum/state contract:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/mirror_checkpoint.py status --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-control-service-smoke-10 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-service-smoke-10/step_000010.pt
```

Expected remote paths:

```text
pose-control-service-smoke-10/full/step_000010.pt
pose-control-service-smoke-10/full/step_000010.pt.complete.json
```

Earlier production coverage also passed:

```bash
PYTHONPATH=. python -m unittest tests.test_production_training tests.test_production_throughput_benchmark -v
PYTHONPATH=. python -m py_compile pose_controlnet/production_training.py scripts/train_production.py tests/test_production_training.py
PYTHONPATH=. python -m unittest tests.test_production_training tests.test_train_mechanics tests.test_pose_reward_wandb -v
```

The focused CPU/no-network tests cover the locked CLI/defaults, batch 32,
200-step warmup, exact LR/pose recipe, loader4 defaults, disabled runtime
alternatives, scientific resume identity, default-off W&B/HF behavior, W&B
entity propagation and checkpoint run-ID continuity, exact 500-step HF milestones, nonfatal HF
failure/local authority, temp-checkpoint rejection, and cumulative-counter
continuity.

## Exact corrected 10-step service-smoke command (do not run from Codex)

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --train-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents_768 --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --pose-sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3_768 --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints --run-name pose-control-service-smoke-10 --max-steps 10 --save-every 5 --diagnostics-every 1 --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-420 --wandb-name pose-control-service-smoke-10
```

## Exact 3000-step operator commands (do not run from Codex)

```bash
cd /home/ubuntu/krea2-pose-controlnet
tmux new-session -d -s pose-production-3000 "cd /home/ubuntu/krea2-pose-controlnet && mkdir -p /lambda/nfs/adhit/krea2-pose/production-logs && set -o pipefail && PYTHONPATH=. python scripts/train_production.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --train-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents_768 --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --pose-sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3_768 --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints --run-name pose-control-production-3000 --max-steps 3000 --save-every 250 --diagnostics-every 50 --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-420 --wandb-name pose-control-production-3000 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 500 2>&1 | tee /lambda/nfs/adhit/krea2-pose/production-logs/pose-control-production-3000.log"

tmux attach -t pose-production-3000
tail -F /lambda/nfs/adhit/krea2-pose/production-logs/pose-control-production-3000.log
watch -n 10 'ls -lh /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000; tail -n 2 /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000/metrics.jsonl'
```

Resume after an interruption:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --train-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents_768 --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --pose-sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3_768 --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints --run-name pose-control-production-3000 --max-steps 3000 --save-every 250 --diagnostics-every 50 --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-420 --wandb-name pose-control-production-3000 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 500 --resume auto
```

## Files changed this session

- `pose_controlnet/production_training.py`
- `pose_controlnet/checkpointing.py`
- `tests/test_train_mechanics.py`
- `tests/test_production_training.py`
- `docs/CODEX_HANDOFF.md`

## Next action

After reviewing this patch, run only the direct synchronous operator mirror
command above against the already-existing `step_000010.pt`, then run its
status command and confirm both remote paths are `true`/`valid_complete: true`.
Do not retrain solely to validate this shutdown fix.
