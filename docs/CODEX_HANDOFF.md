# Project handoff

## Current objective

The final hero-v1 collage correction is complete. The public/review collages
now use uncropped, aspect-preserving artwork and lead with the psychedelic
swordswoman pair. Do not modify frozen release settings, benchmark artifacts,
winner metadata, prompts, training/evaluation code, checkpoints, or evaluation
claims as part of showcase documentation work.

## Definitive public winners

The selection is recorded in
`docs/showcase/final/hero-v1/final_winners.json`:

- `fantasy_mage` — hero variant B (`fantasy_mage_hero_b`)
- `dark_fantasy_jester` — original accepted winner
- `comic_fashion` — hero variant B (`comic_fashion_hero_b`)
- `female_swordswoman_psychedelic` — original accepted winner
- `starry_night_painterly` — hero variant A (`starry_night_painterly_hero_a`)

## Public assets and prompt guidance

- Public collage:
  `docs/showcase/final/hero-v1/final_showcase_collage.png`
- Labeled review collage:
  `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- Reproducible builder: `scripts/build_final_showcase_collage.py`
- Prompt guide: `prompting.md`

The builder reads winner provenance from frozen manifests and copied sidecars.
It only composes the existing condition and generation images. The final
2560×1440 editorial layout consists exclusively of five attached
`[condition | generation]` pairs, with condition on the left and generation on
the right. The upper-left, first-read, dominant pair is the psychedelic
swordswoman. All panels use aspect-preserving containment on black rather than
crop or distortion, so generation composition and every skeleton keypoint are
preserved. It has no public labels or floating control tiles; the labeled
variant has identical geometry and one small concept label per pair.

`prompting.md` retains its core rule, reusable recipe, control/scale guidance,
frozen examples, conflicting-prompt example, pasteable LLM instruction, and
optional Style-LoRA guidance. Stray SVG rendering/copy artifacts are absent.

## Verification this session

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/build_final_showcase_collage.py --check
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/build_final_showcase_collage.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_build_final_showcase_collage -v
# 2 tests passed: frozen winner/assets contract and frozen prompt identity
```

Additional local verification confirmed both output PNGs exist, are RGB
2560×1440 landscape images. Both were visually inspected: all five pairs are
attached and condition-left; the swordswoman is first and upper-left; source
content is fully visible without distortion; starry-night retains its full
figure and sky; mage, comic, swordswoman, and jester retain their intended
composition. `git diff --check` passes.

## Files changed this session

- `scripts/build_final_showcase_collage.py`
- `docs/showcase/final/hero-v1/final_showcase_collage.png`
- `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- `docs/CODEX_HANDOFF.md`

No GPU-dependent work remains. No commit or push was performed.

## Next recommended action

Review the corrected public collage, then stage and commit the existing
repository changes when ready.
