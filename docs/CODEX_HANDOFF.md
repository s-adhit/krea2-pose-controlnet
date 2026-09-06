# Project handoff

## Current objective

The frozen final public hero-v1 showcase package has received its final visual
cleanup. The public/review collages are ready for the README showcase update.
Do not modify frozen release settings, benchmark artifacts, winner metadata,
training/evaluation code, checkpoints, or evaluation claims as part of that
documentation work.

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
It only performs crop/resize/composition on the existing condition and
generation images. The final 2560×1440 editorial layout consists exclusively
of five attached `[condition | generation]` pairs: condition on the left,
generation on the right, with matched panel framing. It has no public labels,
no floating control tiles, and 1.43% exposed dark gutter area. The labeled
variant has the identical geometry and one small concept label per pair.

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
2560×1440 landscape images, and have a 1.43% uncovered canvas fraction;
`prompting.md` has no `[svg]` or `**svg**` artifacts; all repository-relative
Markdown links in the guide resolve; and `git diff --check` passes. Both PNGs
were visually inspected for attached pairs, hierarchy, labels, and whitespace.

## Files changed this session

- `scripts/build_final_showcase_collage.py`
- `docs/showcase/final/hero-v1/final_showcase_collage.png`
- `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- `prompting.md`
- `docs/CODEX_HANDOFF.md`

No GPU-dependent work remains. No commit or push was performed.

## Next recommended action

Review the final public collage in the README showcase update, then stage and
commit the existing repository changes when ready.
