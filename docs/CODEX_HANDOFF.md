# Project handoff

## Current objective

Close the GPU/showcase phase by generating the frozen final native hero-v1
package on the GH200, reviewing it, and copying the package into the docs tree.
Do not train, modify the frozen release contract, change benchmarks, add a
Style-LoRA, commit, or push from Codex.

## Frozen release and winner sources

- Release: `docs/evaluation/release/final_release_v1.json`
  SHA-256 `9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`.
  Canonical runtime is mix-025, Krea-2 Turbo 8, CFG 0, mu 1.15 with
  resolution-dependent mu disabled, control scale 1.0, native geometry, and
  no Style-LoRA.
- Batch 1 source manifest:
  `docs/showcase/final/batch1-v1/final_showcase_batch1_v1.json`
  SHA-256 `629f3b87873b4a5a0bd0306ff6ffd1b9ac04e584976bb856226ab9311417c91b`.
  Accepted winners: `fantasy_mage_m1` (generation
  `02_fantasy_mage_m1.json`, seed 1847302951),
  `dark_fantasy_jester_unique` (`03_dark_fantasy_jester_unique.json`,
  3028147759), and `comic_fashion_unique`
  (`09_comic_fashion_unique.json`, 2519074836).
- Batch 2 source manifest:
  `docs/showcase/final/batch2-v1/final_showcase_batch2_v1.json`
  SHA-256 `b152c97e29a9094260e860df1accdf66814db4898518313ba64300e1715250ae`.
  Accepted winners: `02_female_swordswoman_psychedelic_s2`
  (`04_02_female_swordswoman_psychedelic_s2.json`, 7194308222) and
  `05_starry_night_painterly_s1`
  (`09_05_starry_night_painterly_s1.json`, 7194308251).

## Hero-v1 implementation

- Frozen hero manifest: `docs/showcase/final_hero_showcase_v1.json`,
  SHA-256 `98257de3edae550164bd4a8ecc00934ba8b8885a03d2be8a1f13a6b064c46246`.
  It references the frozen source rows/sidecars instead of restating prompts,
  controls, prepared geometry, or release settings.
- Runner: `scripts/final_hero_showcase.py`. It validates the release and both
  source manifest hashes, resolves the exact winner prompt/control/native
  bucket from them, validates original inference provenance, calls only
  `inference.py`, resumes only when an existing output and sidecar are both
  valid, and fails closed otherwise.
- Output root: `/lambda/nfs/adhit/krea2-pose/showcase/final/hero-v1`.
  New images are `generations/*.png`, their normal inference sidecars remain
  adjacent, compact provenance is `hero_provenance.json`, and the 5x4 review
  sheet is `review_contact_sheet.png`.
- Docs package destination: `docs/showcase/final/hero-v1/`. The copy mode
  copies the manifest, review sheet, summary, five conditions, five original
  winners plus their canonical sidecars, and ten hero variants plus inference
  sidecars. It refuses conflicting destination files.

## Exact new rows/seeds

```text
fantasy_mage_hero_a                    7194308301
fantasy_mage_hero_b                    7194308302
dark_fantasy_jester_hero_a             7194308311
dark_fantasy_jester_hero_b             7194308312
comic_fashion_hero_a                   7194308321
comic_fashion_hero_b                   7194308322
female_swordswoman_psychedelic_hero_a  7194308331
female_swordswoman_psychedelic_hero_b  7194308332
starry_night_painterly_hero_a          7194308341
starry_night_painterly_hero_b          7194308342
```

All ten are deterministic and the runner rejects a collision with any frozen
Batch 1/2 seed.

## GH200 commands

Run from the actual GH200 host with the NFS controls, original winners,
checkpoint endpoints, and Turbo checkpoint mounted:

```bash
cd /home/ubuntu/krea2-pose-controlnet && PYTHONPATH=. UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_hero_showcase.py preflight
cd /home/ubuntu/krea2-pose-controlnet && PYTHONPATH=. UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_hero_showcase.py generate --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors
cd /home/ubuntu/krea2-pose-controlnet && PYTHONPATH=. UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_hero_showcase.py review-sheet
cd /home/ubuntu/krea2-pose-controlnet && PYTHONPATH=. UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_hero_showcase.py copy-to-docs
```

`generate` produces only the ten new hero images; a valid existing image plus
its matching inference sidecar is skipped. `review-sheet` also writes the
compact summary. `copy-to-docs` regenerates/validates the review package
before copying it.

## Verification this session

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/final_hero_showcase.py tests/test_final_hero_showcase.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_final_hero_showcase tests.test_inference tests.test_final_release_decision -v
# 22 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_hero_showcase.py --help
sha256sum docs/showcase/final_hero_showcase_v1.json docs/evaluation/release/final_release_v1.json
git diff --check
```

No images were generated, and no frozen release, training, or benchmark files
were changed. The actual `preflight` requires the mounted GH200/NFS artifacts
and was deliberately not run in the Codex sandbox.

After hero-v1 is generated and pushed, no remaining pre-blog task requires
the GH200 GPU.
