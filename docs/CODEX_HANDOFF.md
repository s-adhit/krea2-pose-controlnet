# Project handoff

## Current objective

The final public showcase collage and README presentation are complete. Do not
modify frozen winner selection, prompts, hero-generation assets, prompting
guide, release contract, benchmarks, training code, or evaluation code as part
of this completed showcase milestone.

## Final public showcase

- Public collage: `docs/showcase/final/hero-v1/final_showcase_collage.png`
- Labeled review collage:
  `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- Frozen winner contract: `docs/showcase/final/hero-v1/final_winners.json`
- Reproducible builder: `scripts/build_final_showcase_collage.py`

The builder creates a dense 2449×1368 landscape composition with five
inseparable horizontal `[condition | generation]` pairs. Normal reading order
begins with the psychedelic female swordswoman. Every condition has the same
displayed 200×600 bounding tile and is fully contained on black. Every
generation uses native-aspect containment with no crop or distortion. The
jester has a substantial 760-pixel-high panel, while the other generations
are 600 or 760 pixels high according to the two-row native-aspect packing;
none is relegated to a thumbnail. The labeled version uses identical geometry
and adds only small concept labels.

## README

`README.md` now has a concise `## Showcase` section immediately after the
introduction. It presents only the final public collage and links to
`prompting.md`. The prior Status section, stale individual showcase examples,
review-sheet links, and interim showcase prose were removed. Other inference,
evaluation, and benchmark material was left unchanged.

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
confirmed as RGB 2449×1368 landscape images and visually inspected: five
concepts appear exactly once; each condition is attached on the left of its
generation; all condition tiles share dimensions; skeletons and native
generation compositions are fully visible; and no large unused canvas region
is present. README was checked for the new public image path and for the
absence of Status / Project Status / Current Status headings.

## Files changed this session

- `README.md`
- `scripts/build_final_showcase_collage.py`
- `tests/test_build_final_showcase_collage.py`
- `docs/showcase/final/hero-v1/final_showcase_collage.png`
- `docs/showcase/final/hero-v1/final_showcase_collage_labeled.png`
- `docs/CODEX_HANDOFF.md`

No GPU-dependent work remains. No commit or push was performed.

## Next recommended action

Review the final collage and README, then stage and commit the existing
changes when ready.
