# Project handoff

## Current objective

Hero-v2 has been frozen and packaged for human visual review. Do not alter its
winner list, frozen evaluation artifacts, hero-v1 provenance, release contract,
or release checkpoint identity without an explicit new decision.

## Frozen hero-v2 winners

The authoritative manifest is `docs/showcase/final/hero-v2/final_winners.json`.
Its order is also the left-to-right order of both final montages:

1. psychedelic swordswoman
2. fantasy mage
3. comic fashion
4. starry-night painterly
5. dark-fantasy jester
6. elegant male warrior / wandering knight (`canonical-v1`)
7. gothic masked noble with attendant (`canonical-v1`)
8. painterly mythic companions (`retry-a`)
9. moonlit lotus princess (`retry-b`)
10. astral empress / cosmic oracle (`retry-b`)

Explicitly excluded from the final hero are realistic female warrior, realistic
fashion/editorial portrait, original moonlit priestess/dreamy floral oracle,
and stained-glass saint/celestial figure.

## Final presentation assets

- Generation montage:
  `docs/showcase/final/hero-v2/final/final_generation_montage.png`
- Matching condition montage:
  `docs/showcase/final/hero-v2/final/final_condition_montage.png`
- Stable numbered generation/condition copies:
  `docs/showcase/final/hero-v2/final/`

`scripts/package_hero_v2_final.py` deterministically copies the selected,
already-recorded source images and controls, records prompt/seed/geometry/
candidate/control-scale provenance plus SHA-256s in the frozen manifest, and
builds the two same-order justified montages. It does not generate images.

## Documentation refreshed

- `README.md` now presents the hero-v2 generation montage and links the
  matching conditions.
- `docs/release/HF_MODEL_CARD.md` uses the same montage through the raw GitHub
  URL.
- `prompting.md` preserves the geometry-versus-appearance guidance and now has
  the final ten concise, exact-prompt examples with their condition/generation
  images, seed, native dimensions, candidate, and control scale.

## Verified release/environment facts

- Release candidate: `mix-025`; release ID: `krea2-pose-control-lora-v1`.
- Release SHA-256:
  `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Runtime contract: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, native geometry,
  no Style-LoRA. Most heroes use control scale `1.0`; painterly mythic
  companions uses `1.25`.
- GH200 host verification remains Linux ARM64, approximately 96 GB HBM,
  Python 3.10.12, PyTorch 2.7.0 / CUDA 12.8 / cuDNN 9.8 / Triton 3.3.0;
  BF16, SDPA, and `torch.compile` passed from the normal host shell. Use `uv`.

## Files changed this session

- `scripts/package_hero_v2_final.py`
- `docs/showcase/final/hero-v2/final_winners.json`
- `docs/showcase/final/hero-v2/final/` (20 selected source copies and two
  montages)
- `README.md`
- `docs/release/HF_MODEL_CARD.md`
- `prompting.md`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
uv run python scripts/package_hero_v2_final.py
# visual inspection of both final montages
```

Before ending, run `git diff --check` and `git status --short`. Do not commit
or push.

## Next recommended action

Conduct human visual review of the two final montages, then commit/push the
approved package and upload the refreshed HF README/model card.
