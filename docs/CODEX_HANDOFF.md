# Phase 1 handoff

## Current state

The checkpoint-mirroring milestone is complete; no training was launched and no CUDA memory was allocated. The concurrent PCK/provenance work remains independent: source-derived COCO references are the only currently valid route to genuine PCK. Do not modify evaluation/PCK code while working on training mechanics unless strictly necessary.

## Checkpoint mirror semantics

- `--hf-mirror-every-steps N` defaults to `0`, which leaves the existing wall-clock-only HF mirroring behavior unchanged.
- With `N > 0`, an exact local full checkpoint is enqueued after its atomic save when `global_step % N == 0`. The CLI rejects negative values, missing `--hf-repo-id`, non-positive `--save-every`, and a cadence not divisible by `--save-every`.
- Step requests require canonical identity: local `step_S.pt` uploads to `run_name/full/step_S.pt` and its existing SHA-256 completion marker. A FIFO background queue retains each exact request while another upload is running; it cannot be replaced by a later checkpoint.
- Timed and step requests are independent. Duplicate paths are suppressed while queued and after success during the process lifetime. `hf/mirror_reason` records `step` or `timed` without altering W&B step ordering.
- Local pruning preserves every queued/in-flight upload source. On successful upload it still retains the newest two normal valid local checkpoints, plus all queued sources. Failed requests become eligible for normal later retention after their bounded retries finish.
- Existing marker-backed HF recovery and exact-step validation remain unchanged.

For `--save-every 25 --hf-mirror-every-steps 100`, a resume at step 500 selects precisely 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, and 1500 for guaranteed mirror attempts, assuming those steps complete and their uploads succeed. Non-100 local saves are not step-selected.

## PCK provenance status

- The PoseBridge snapshot lacks source pose annotations. COCO stems can be joined exactly to official COCO 2017 keypoint annotations; Human-Art mapping remains unverified pending licensed annotations; Danbooru has no compatible authoritative source keypoints.
- `pose_controlnet/reference_pose.py`, `scripts/build_coco_reference_pose.py`, and `tests/test_reference_pose.py` are existing, uncommitted concurrent-work files. The COCO sidecar command from the prior handoff remains the next PCK-specific action.

## Files changed in this checkpoint session

- `train.py`
- `pose_controlnet/config.py`
- `pose_controlnet/checkpointing.py`
- `pose_controlnet/wandb_logging.py`
- `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

## Verification

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics tests.test_wandb_logging` (39 tests). The expected argparse error output is exercised by validation tests.
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile train.py pose_controlnet/checkpointing.py pose_controlnet/config.py pose_controlnet/wandb_logging.py tests/test_train_mechanics.py`
- PASS: `git diff --check`

## Exact next host command

Set `HF_REPO_ID` to the intended private HF model repository, then run the bounded 500 -> 1500 continuation (do not run until separately authorized):

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py --run-name pose-learning-500 --max-steps 1500 --allow-extended-training --microbatch-size 1 --gradient-accumulation-steps 32 --resume auto --save-every 25 --hf-repo-id "${HF_REPO_ID:?set the private HF model repo id}" --hf-mirror-every-steps 100 --hf-mirror-every-seconds 3600
```

Before launching, confirm that microbatch `1` remains the intended profiled GH200 setting and that the specified HF repository is private and authenticated. This command does not alter optimizer/LR/warmup/LoRA/compile/GC/data/text-conditioning defaults.
