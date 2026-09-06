# Project handoff

## Current objective

The final public showcase collage has completed its final simplification pass.
Do not modify frozen winner selection, prompts, hero-generation assets,
prompting guide, release contract, benchmarks, training code, or evaluation
code as part of this completed showcase milestone.

## Final public showcase

- Public collage: `docs/showcase/final/hero-v1/final_showcase_collage.png`
- Labeled review collage:
  `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- Frozen winner contract: `docs/showcase/final/hero-v1/final_winners.json`
- Reproducible builder: `scripts/build_final_showcase_collage.py`

The builder creates a dense 4240×1808 landscape composition in a rectangular
3-over-2 layout. Normal reading order is psychedelic female swordswoman,
fantasy mage, comic fashion, starry-night painterly, then dark-fantasy jester.
Every concept is a strict `[condition | generation]` pair with exactly equal
displayed condition and generation panel dimensions. The upper three pairs use
700×1000 panels; the lower two use 1054×800 panels, keeping their areas about
20% above the upper pairs while filling the rectangle cleanly. Native-aspect
containment on black preserves every skeleton and generation pixel without
distortion or crop. The jester is no longer a dominant anchor. The labeled
version has identical geometry and only adds small concept names.

## README

`README.md` already has a concise `## Showcase` section immediately after the
introduction. It presents the final public collage and links to `prompting.md`;
it was not changed in this pass. Other inference, evaluation, and benchmark
material remains unchanged.

## Verification this session

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py --check
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/build_final_showcase_collage.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_build_final_showcase_collage -v
git diff --check
```

The focused collage tests pass (3 tests): frozen winner/assets contract,
frozen prompt identity, and equal-weight geometry/order. Both output PNGs were
confirmed as RGB 4240×1808 landscape images and visually inspected: five
concepts appear exactly once in the stated order; each condition is left of its
generation with identical displayed dimensions; skeletons and native generation
compositions are fully visible; the tall starry-night composition is preserved;
the jester is comparable to the other pairs; and no large unused canvas region
is present. README was checked for its correct public image path.

## Files changed this session

- `scripts/build_final_showcase_collage.py`
- `tests/test_build_final_showcase_collage.py`
- `docs/showcase/final/hero-v1/final_showcase_collage.png`
- `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- `docs/CODEX_HANDOFF.md`

No GPU-dependent work remains. No commit or push was performed.

## Next recommended action

Review the final collage, then stage and commit the existing changes when
ready.
