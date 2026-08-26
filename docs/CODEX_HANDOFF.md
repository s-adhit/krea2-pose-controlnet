# Phase 1 handoff

## Current objective

The two pre-100-step milestones are complete: optimizer-step warmup semantics and full Hugging Face checkpoint mirroring/recovery. Do not run the 100-step job or a production run without explicit authorization.

## Decisions in force

- Krea-2 Raw, clean skeleton-control latent channel concatenation, rank-64 LoRA, BF16 flow-matching MSE, seed 42, frozen backbone.
- Verified Gate-F runtime: MB2, accumulation 16, effective batch 32, six main transformer blocks checkpointed, compile off, cached text conditioning.
- AdamW remains `lr=1e-4`, betas `(0.9, 0.99)`, weight decay `0`; warmup is 200 optimizer updates. Update N now actually uses `min(1, N/200)*1e-4`.
- Canonical local/NFS checkpoint root remains `/lambda/nfs/adhit/krea2-pose/checkpoints`; it is authoritative. HF is a configurable, private, best-effort mirror using the ambient authenticated HF account—no credentials are placed in config, checkpoint, or telemetry.

## Completed gates/findings

- Gates A–E remain PASS. The earlier 10-step gradient diagnostic regression is fixed but the real GH200 repeat remains an authorized future action.
- `OptimizerStepWarmup` installs the rate for the impending optimizer update, and resume installs the rate for the next update. Training telemetry records that pre-update rate rather than the following update's rate.
- Full `.pt` training checkpoints are atomically written, deserialized, and schema-validated before publication. Required state includes trainable model, AdamW, scheduler, global/epoch/batch position, Python/NumPy/Torch/CUDA RNG, flow generator, and configuration.
- `--hf-repo-id` enables a background full-checkpoint mirror; default cadence is 3600 seconds and `--hf-mirror-every-seconds` permits short host tests. Upload uses retry/backoff and failures are non-fatal. A remote checkpoint is resumable only after its upload and a checksum completion marker both exist.
- `--resume auto` validates newest local `.pt` first; only if absent does it enumerate completion-marked remote files, download, checksum, deserialize, and validate state before selection. Corrupt/newest remote candidates are skipped. Two newest valid local full checkpoints are retained after a successful mirror.
- HF telemetry records success, uploaded step, age of the last remote success, and credential-redacted error status in local JSONL/W&B.

## Files changed this session

- `train.py`
- `pose_controlnet/checkpointing.py`
- `pose_controlnet/config.py`
- `pose_controlnet/wandb_logging.py`
- `tests/test_train_mechanics.py`
- `tests/test_wandb_logging.py`
- `docs/CODEX_HANDOFF.md`

## Tests run

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics tests.test_wandb_logging`; `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile train.py pose_controlnet/checkpointing.py pose_controlnet/config.py pose_controlnet/wandb_logging.py tests/test_train_mechanics.py tests/test_wandb_logging.py`; and `git diff --check`.

Focused tests cover warmup updates 1/10/200, resumed warmup, rate observed by the optimizer, atomic full-state recovery, successful HF marker upload, non-fatal retry/backoff, local-first recovery, remote fallback, and corrupt remote candidate rejection. No real training, remote upload, commit, or push was performed.

## Exact next recommended action

From the GH200 host, authenticate with `hf auth whoami`, then run the short HF recovery test specified in the current user request. Use `--max-steps 1`, `--save-every 1`, and `--hf-mirror-every-seconds 0`; move only that test run's local `.pt` aside; then resume the same run with `--max-steps 2 --resume auto`. The output must report `[resume] loaded validated full checkpoint ... at optimizer step 1`. Stop after that bounded recovery proof.

After explicit authorization, the (not yet run) 100-step command is MB2, accum16, GC6, compile off, cached text conditioning, and effective batch 32.
