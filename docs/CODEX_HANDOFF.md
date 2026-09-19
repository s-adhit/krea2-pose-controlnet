# Project handoff

## Current objective

Materialize the approved three-concept `hero-v2` retry-b batch on the writable
GH200 host, then conduct human visual review. Do not modify `canonical-v1`,
the accepted keepers, `retry-a`, hero-v1, or evaluation artifacts.

## Hero-v2 retry-b status

- Manifest: `docs/showcase/final/hero-v2/retries/retry-b/retry_b.json`
- Readable plan: `docs/showcase/final/hero-v2/retries/retry-b/RETRY_B.md`
- Runner: `scripts/generate_hero_v2_retry.py --batch retry-b`; the shared
  runner retains retry-a's default behavior, has explicit per-batch manifest
  and output-root validation, and reuses canonical release/no-overwrite helpers
  plus `inference.py` for runtime and generation.
- NFS output: `/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/retry-b/`
- Repo presentation output:
  `docs/showcase/final/hero-v2/generations/retry-b/`
- Concepts: realistic female warrior (replacement 1216x832 control,
  seed 7194308601, scale 1.25); moonlit lotus princess (768x1344,
  seed 7194308602, scale 1.0); astral empress / cosmic oracle (1024x1024,
  seed 7194308603, scale 1.0).
- The runner only creates the batch-specific output root and refuses differing
  existing files. This sandbox reports the NFS showcase root as non-writable,
  so no generation was attempted here.

Run on the writable GH200 host:

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/generate_hero_v2_retry.py preflight --batch retry-b
uv run python scripts/generate_hero_v2_retry.py generate --batch retry-b
```

## Verified release and environment facts

- GH200 host: Linux ARM64, approximately 96 GB HBM, Python 3.10.12,
  PyTorch 2.7.0 / CUDA 12.8 / cuDNN 9.8 / Triton 3.3.0; BF16, SDPA, and
  `torch.compile` were verified from the normal host shell. Use `uv`; do not
  replace the working CUDA/PyTorch stack.
- Release artifact:
  `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors`
- Release SHA-256:
  `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`
- Runtime contract: `mix-025`, Krea-2 Turbo, 8 steps, CFG 0, mu 1.15,
  resolution-dependent mu disabled, no Style-LoRA, native geometry.

## Relevant prior state

- `canonical-v1` was generated once; its summary is at
  `docs/showcase/final/hero-v2/generations/canonical-v1/GENERATION_SUMMARY.md`.
- `retry-a` remains intact and passes its own preflight. Its plan and manifest
  are under `docs/showcase/final/hero-v2/retries/retry-a/`.
- The audition is the authoritative control provenance:
  `docs/showcase/final/hero-v2/control-audition/audition_candidates.json`.

## Files changed this session

- `scripts/generate_hero_v2_retry.py`
- `docs/showcase/final/hero-v2/retries/retry-b/retry_b.json`
- `docs/showcase/final/hero-v2/retries/retry-b/RETRY_B.md`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
uv run python -m py_compile scripts/generate_hero_v2_retry.py
uv run python scripts/generate_hero_v2_retry.py preflight --batch retry-b
uv run python scripts/generate_hero_v2_retry.py preflight --batch retry-a
```

Before ending, run `git diff --check` and `git status --short`. Do not commit,
push, or launch production training.
