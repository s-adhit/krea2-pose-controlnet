# Phase 1 handoff

## Current bounded objective and status

The repository is prepared for one LR-only continuation ablation. No training,
checkpoint download, checkpoint write, W&B run, HF upload, commit, or push was
started in this session.

## Exact 900 -> 1500 LR branch

- Source is strictly the completion-marked, SHA-256-validated and full-state
  deserialized HF archive
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`, namespace
  `pose-learning-1500/full/step_000900.pt`. Its embedded `global_step` must be
  exactly `900`; no local/latest/nearest/timed replacement is accepted.
- Launch mode is `--lr-branch-900-to-1500`. It derives the complete
  `TrainConfig` from the source checkpoint and rejects missing/extra config
  fields, wrong run, wrong base LR, wrong target, pre-warmup state, insufficient
  checkpoint cadence, or a different HF repository. Consequently all model,
  data, precision, batch, sampling, dropout, optimizer, diagnostics, validation,
  checkpoint, and mirror settings remain source-identical.
- It restores trainable model state, AdamW state including first/second moments,
  warmup scheduler state/progress, `global_step`, epoch, batch position, Python/
  NumPy/torch/CUDA RNG, and flow generator state before the branch override.
- The scheduler is warmup-only. At restored step 900 its counter stays 900;
  warmup is never restarted. The branch sets both optimizer group LR and the
  scheduler's restored `base_lrs` to exactly `5e-5`, preventing the next
  `scheduler.step()` from restoring `1e-4`. An assertion runs after every branch
  optimizer/scheduler update.
- Target is global step `1500` (exactly optimizer steps 901 through 1500).
  Source checkpoint cadence must preserve steps 1000, 1100, 1200, 1300, 1400,
  and 1500.

## Isolated branch namespace

- W&B run name: `pose-learning-900-lr5e5-to1500` in the source-configured
  project/entity (expected `adhit-projects/Krea-2-PoseControl-Lora`).
- Local run/checkpoint root:
  `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500`
- Local metrics:
  `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/metrics.jsonl`
- HF branch namespace:
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints/pose-learning-900-lr5e5-to1500/full/step_00xxxx.pt`
- Source recovery download, if needed, is isolated below
  `.../pose-learning-900-lr5e5-to1500/source-step-900-recovery`; all new
  checkpoints and metrics are guarded from writing to `pose-learning-1500`.

## Verified tests

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics tests.test_evaluation tests.test_turbo_evaluation` (59 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile train.py pose_controlnet/checkpointing.py tests/test_train_mechanics.py tests/test_evaluation.py tests/test_turbo_evaluation.py`.
- New focused coverage proves exact HF step-900 routing, full resume data/RNG/
  flow-generator restoration, AdamW moment preservation, scheduler restoration,
  LR `5e-5` immediately after resume, after the first update/scheduler step
  (step 901), and at later scheduler progress (step 1100), no warmup restart,
  target 1500, source-derived non-LR hyperparameters, and isolated local/HF/
  metrics namespaces.

## Exact future GH200 launch and monitoring commands

Do not run without explicit authorization to start this branch.

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
cd /home/ubuntu/Krea-2-Pose-ControlNet
uv run python train.py --lr-branch-900-to-1500
```

```bash
tail -f /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/metrics.jsonl
watch -n 10 nvidia-smi
uv run python scripts/mirror_checkpoint.py status \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-learning-900-lr5e5-to1500 \
  --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001000.pt
```

The original 1500-run telemetry median was 28.929 seconds/optimizer step
(1.106 samples/sec). For 600 more steps, the training-only estimate is about
4 hours 49 minutes; allow roughly 5 hours including validation, checkpoints,
and HF mirroring.

## Files changed this session

- `train.py`
- `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

## Next action

Wait for explicit authorization before executing the GH200 launch command.
