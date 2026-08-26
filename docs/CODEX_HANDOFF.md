# Phase 1 handoff

## Current objective

Gate D latent-shard creation and hard verification are implemented. The code is ready for the required real-GH200 smoke run, but this Codex audit shell cannot perform that run because it has no CUDA device and cannot reach the Hub to fetch the uncached Qwen VAE weights. No full shard generation was started.

## Decisions in force

- Krea-2 Raw uses `AutoencoderKLQwenImage` from `Qwen/Qwen-Image/vae`; BF16 encoding and paired geometry remain exclusively in the project-owned VAE and paired-preprocessing helpers.
- Canonical immutable snapshot root: `/lambda/nfs/adhit/krea2-pose/posebridge_hf`.
- Full latent output root: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Shards are Torch `.pt` archives containing a format-versioned split and a list of per-sample dictionaries. Each sample has float32 CPU `image_latent`/clean `control_latent`, caption, stem/file identity, split, and paired bucket/geometry metadata. The VAE remains BF16 for compute.
- Default shard size is 256 samples. A paired latent payload is about 2 MiB for typical buckets, making a shard roughly 0.5 GiB: suitable for NFS sequential throughput while keeping retry/recovery granularity practical.
- Shards are written to a same-directory temporary file, flushed/fsynced, loaded and hard-validated, then atomically renamed. Existing final shards are reused only when they validate against their deterministic split/stem range. Temporary files are never considered complete.

## Completed/green checks

- Gate B physical resolution and Gate C/C2 GH200 VAE smoke: host-verified PASS by the user (Qwen VAE, BF16, landscape/portrait/square, matched finite nonzero controls).
- Shard unit and helper regression tests: PASS (21 tests): `.venv/bin/python -m unittest -v tests/test_shards.py tests/test_dataset_index.py tests/test_paired_preprocessing.py tests/test_vae_preprocessing.py`
- Syntax checks: PASS — `.venv/bin/python -m py_compile prepare_shards.py scripts/verify_shards.py`.
- `git diff --check`: PASS.

## Smoke status

- Attempted a three-sample real-data smoke (`--max-samples-per-split 1`) in a unique `/tmp/posebridge-latent-smoke.*` root.
- BLOCKED before encoding: this audit shell reported `cuda_available=False` and Hub DNS failed while the VAE weights were not available locally. It did read the mounted snapshot/manifests. No latent shard was created; the incomplete temporary output was safely removed.
- `scripts/verify_shards.py --allow-partial` was run against that incomplete smoke root and correctly rejected it with `No shard files found`.

## Files changed this session

- `prepare_shards.py`
- `scripts/verify_shards.py`
- `tests/test_shards.py`
- `docs/CODEX_HANDOFF.md`

## Current blockers

- Run the tiny smoke from the real GH200 production shell, where CUDA and the already host-verified Qwen VAE artifact are available. This is an audit-shell access issue, not a data or shard-format failure.

## Exact next action

First run and inspect the bounded smoke (then remove its `/tmp` output):

```bash
.venv/bin/python prepare_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /tmp/posebridge-latent-smoke \
  --device cuda --shard-samples 1 --max-samples-per-split 1
.venv/bin/python scripts/verify_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /tmp/posebridge-latent-smoke --allow-partial
```

After explicit authorization, the exact full persistent creation command is:

```bash
.venv/bin/python prepare_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --device cuda --shard-samples 256
```

Then run the hard full gate (must report train 16,503, val 889, diagnostic_val 24, total 17,416):

```bash
.venv/bin/python scripts/verify_shards.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-root /lambda/nfs/adhit/krea2-pose/posebridge_latents
```
