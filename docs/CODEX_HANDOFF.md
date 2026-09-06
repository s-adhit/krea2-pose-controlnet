# Project handoff

## Current objective

The final public hero-v1 showcase package is complete. Do not modify the frozen
release contract, benchmark results, training code, checkpoints, or evaluation
numbers as part of further blog/documentation work.

## Definitive public winners

The selection is materialized and provenance-validated in
`docs/showcase/final/hero-v1/final_winners.json`:

- `fantasy_mage` — hero variant B (`fantasy_mage_hero_b`)
- `dark_fantasy_jester` — original accepted winner
- `comic_fashion` — hero variant B (`comic_fashion_hero_b`)
- `female_swordswoman_psychedelic` — original accepted winner
- `starry_night_painterly` — hero variant A (`starry_night_painterly_hero_a`)

Metadata retains only the approved alternates: fantasy mage original,
comic-fashion original, swordswoman hero A, and starry-night original.

## Public assets and prompt guidance

- Public collage:
  `docs/showcase/final/hero-v1/final_showcase_collage.png`
- Labeled review collage:
  `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- Reproducible builder:
  `scripts/build_final_showcase_collage.py`
- Final LLM-oriented prompt guide: `prompting.md`
- README has a small pointer to the public/review collages, frozen contract,
  and prompt guide.

The builder resolves prompts, seeds, controls, SHA-256 values, sidecars,
candidate/runtime/release fields, and Batch 1/2 source references exclusively
from `docs/showcase/final_hero_showcase_v1.json`, the copied hero-v1 package,
and the frozen Batch 1/2 manifests. It uses PIL resize/crop/padding/composition
only; no source image pixels are generatively edited.

## Frozen sources

- Hero manifest: `docs/showcase/final_hero_showcase_v1.json`, SHA-256
  `98257de3edae550164bd4a8ecc00934ba8b8885a03d2be8a1f13a6b064c46246`
- Batch 1: `docs/showcase/final/batch1-v1/final_showcase_batch1_v1.json`,
  SHA-256 `629f3b87873b4a5a0bd0306ff6ffd1b9ac04e584976bb856226ab9311417c91b`
- Batch 2: `docs/showcase/final/batch2-v1/final_showcase_batch2_v1.json`,
  SHA-256 `b152c97e29a9094260e860df1accdf66814db4898518313ba64300e1715250ae`

## Verification this session

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_build_final_showcase_collage -v
# 2 tests passed: frozen winner/assets contract and guide prompt identity
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py --check
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/build_final_showcase_collage.py tests/test_build_final_showcase_collage.py
git diff --check
```

Both PNG outputs were opened and verified as valid 2800×1900 RGB images. No
GPU workload was launched. No frozen release, benchmark, training, checkpoint,
or evaluation artifact was changed.

## Files changed this session

- `scripts/build_final_showcase_collage.py`
- `tests/test_build_final_showcase_collage.py`
- `docs/showcase/final/hero-v1/final_winners.json`
- `docs/showcase/final/hero-v1/final_showcase_collage.png`
- `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- `prompting.md`
- `README.md`
- `docs/CODEX_HANDOFF.md`

All GPU-dependent pre-blog work is complete. The GH200 can be terminated once
the current repository changes are committed and pushed.
