# Project handoff

## Current objective and live status

The only active capacity task was fixing the COCO-32 save crash and providing
an exact, fail-closed resume path. It is implemented and CPU-tested. Codex did
**not** train, evaluate, alter existing checkpoints, delete run outputs, commit,
or push.

The live `overfit32-coco-r64-mse` run completed optimizer step 150, then
crashed before publishing a step-150 checkpoint. Root cause: the trainer used
`global_step % 50 == 0` while `per_step_exposures()` correctly permits only
scientific milestones `0, 50, 100, 200, 300, 400, 500`.

## Scientific/architecture contract (unchanged)

- Fresh Krea-2 Raw base; fresh rank/alpha 64/64 Pose-Control LoRA, existing
  28 × 8 LoRA targets, and `ControlInputLayer` only.
- Flow-MSE only; no pose reward/critic; LR `1e-4` constant; warmup 0.
- Microbatch 1 × accumulation 8 = effective batch 8; exact immutable COCO-32
  manifest; deterministic cached-text 10% dropout; no spatial augmentation.
- Terminal optimizer step remains exactly 500. Production `train.py` behavior
  was not changed.

## Save schedule and resume behavior

`OVERFIT_CHECKPOINT_STEPS` is now the sole trainer save authority:
`50, 100, 200, 300, 400, 500`. Step 0 remains the fresh-model reference
checkpoint/evaluation point. Steps 150/250/350/450 cannot trigger a save.
Every checkpoint save additionally asserts that its step is in the complete
scientific list including step 0.

`scripts/train_overfit_capacity.py --resume PATH` is explicit only; normal
invocation remains a fresh-LoRA run and still refuses an existing run directory.
Resume only accepts a checkpoint in the named experiment's own directory and
requires its metadata, exact manifest stems, experiment name, R64/224-target
architecture audit, flow-MSE-only provenance, zero warmup, exact LR/config,
authoritative embedded step/name, scheduler state, exact deterministic
32-sample epoch/batch position, exposure accounting, and full checkpoint
schema. It restores trainable model state, AdamW state, scheduler, global
step, epoch/batch position, Python/NumPy/torch RNG, CUDA RNG when available,
and flow-generator RNG before more work begins.

Read-only audit of
`/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-coco-r64-mse`
found valid `step_000000.pt`, `step_000050.pt`, and `step_000100.pt` only.
The latest valid exact-resume checkpoint is `step_000100.pt` (2,586,424,412
bytes); its embedded progress is `global_step=100`, `epoch=24`,
`batch_position=32`.

Exact GH200 operator command (do not run from Codex):

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_overfit_capacity.py \
  --experiment overfit32-coco-r64-mse \
  --resume /lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-coco-r64-mse/step_000100.pt
```

Completion checkpoint set: `step_000000.pt`, `step_000050.pt`,
`step_000100.pt`, `step_000200.pt`, `step_000300.pt`, `step_000400.pt`, and
`step_000500.pt`; no 150/250/350/450 checkpoints.

## Metrics and W&B

The live `metrics.jsonl` has steps 1–150. On resume from step 100, the harness
preserves the full pre-resume file verbatim as
`metrics.pre_resume_after_step_000100.jsonl`, atomically retains only steps
1–100 in authoritative `metrics.jsonl`, then appends re-executed 101–500.
This prevents duplicate authoritative step records without discarding evidence
of the interrupted tail. W&B run ID was not persisted by the original live
run, so safe reuse cannot be proven: default W&B logging starts a continuation
segment with the same display name. Local `metrics.jsonl` is authoritative.

## Files changed and checks

- `pose_controlnet/overfit_capacity.py`: authoritative nonzero save schedule
  helper and contract assertion.
- `scripts/train_overfit_capacity.py`: explicit fail-closed exact resume,
  state restoration, metrics reconciliation, and schedule-only saves.
- `tests/test_overfit_capacity.py`: focused schedule, terminal, resume-state,
  identity/manifest/non-overfit rejection, fresh-start, metrics, and scientific
  contract tests.
- `docs/CODEX_HANDOFF.md`: this state.

PASS:

```bash
python -m unittest tests.test_overfit_capacity -v  # 10 tests
python -m unittest tests.test_train_mechanics -v   # 42 tests
python -m py_compile scripts/train_overfit_capacity.py pose_controlnet/overfit_capacity.py tests/test_overfit_capacity.py
```

Next action: operator may run the exact explicit resume command above. Do not
start evaluation from Codex; do not run a fresh COCO capacity command against
the existing checkpoint namespace.
