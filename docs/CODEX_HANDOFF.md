# Phase 1 handoff

## Current objective

Gate D shard completion/resume correctness and preprocessing observability are implemented. The next bounded action is a real-GH200 interrupted-resume smoke; do not start the full 17,416-sample latent run without authorization.

## Decisions in force

- Krea-2 Raw uses `AutoencoderKLQwenImage` from `Qwen/Qwen-Image/vae`; BF16 encoding and paired geometry remain exclusively in the project-owned VAE and paired-preprocessing helpers.
- Canonical immutable snapshot root: `/lambda/nfs/adhit/krea2-pose/posebridge_hf`; full latent output root: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Shards are format-versioned Torch `.pt` archives with float32 CPU image/control latents, immutable identity/caption, and paired bucket/geometry metadata. Default shard size is 256.
- `shards.json` is only a run description until atomically rewritten with `complete=true` after full physical validation. It is never proof of completion on restart or verification.

## Completed/green checks

- Gate B physical resolution and Gate C/C2 GH200 VAE smoke: host-verified PASS by the user (Qwen VAE, BF16, landscape/portrait/square, matched finite nonzero controls).
- Completion/resume fix: PASS. Each run atomically records `complete=false` before VAE load or shard creation, removes only tool-created stale temporary files, validates/reuses valid final shards, and atomically replaces missing/corrupt planned shards. The VAE is loaded only if recomputation is actually needed.
- A final completion manifest is written only after deterministic per-shard validation and exact aggregate physical membership validation. Full counts must be train 16,503, val 889, diagnostic_val 24, total 17,416, with no missing or duplicate samples.
- Verification rejects malformed/mismatched metadata, missing/extra shards, wrong physical membership/counts, and the discovered `complete=true` plus zero-shard state.
- Live preprocessing reports split, completed/total samples, shard, samples/s, elapsed, ETA, and reused shard status.
- Targeted shard tests: PASS (6 tests): `.venv/bin/python -m unittest -v tests/test_shards.py`.
- Syntax checks: PASS — `.venv/bin/python -m py_compile prepare_shards.py scripts/verify_shards.py`.
- `git diff --check`: PASS.

## Files changed this session

- `prepare_shards.py`
- `scripts/verify_shards.py`
- `tests/test_shards.py`
- `docs/CODEX_HANDOFF.md`

## Current blocker

- This audit shell cannot perform the GH200/CUDA shard smoke. No full latent generation was started.

## Exact next action

From the real GH200 production shell, prove interruption/restart on a disposable output: interrupt the first command after its first progress line, then run it again.

```bash
.venv/bin/python prepare_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /tmp/posebridge-latent-resume-smoke \
  --device cuda --shard-samples 1 --max-samples-per-split 2
.venv/bin/python prepare_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /tmp/posebridge-latent-resume-smoke \
  --device cuda --shard-samples 1 --max-samples-per-split 2
.venv/bin/python scripts/verify_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /tmp/posebridge-latent-resume-smoke --allow-partial
```

Expected: interruption leaves `complete=false`; restart reports existing valid shards as `(reused)`; verification reports two samples per split. Remove the disposable `/tmp/posebridge-latent-resume-smoke` output after inspection. The full persistent creation command remains blocked pending explicit authorization.
