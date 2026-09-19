# Project handoff

## Current objective

Complete the control-only audition for the seven new `hero-v2` concepts.
No final image was generated, selected, or promoted; frozen `hero-v1` and
evaluation artifacts were not modified.

## Frozen input verification

PASS before materialization:

- `docs/evaluation/release/final_release_v1.json`:
  `9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`
- parent-4000: `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd`
- finish-control-a4300: `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff`

Both full source states have 450 float32 trainable control/LoRA tensors,
215,488,512 parameters, and identical recorded Krea-2 Raw provenance.

## Release artifact

- Intended public path (blocked only by this sandbox mount):
  `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors`
- Verified staged artifact:
  `/tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors`
- Adjacent provenance JSON:
  `/tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors.provenance.json`
- SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`
- Format: `safetensors`; direct model tensors only (no optimizer, scheduler,
  RNG, counters, or unrelated full-training state).
- Tensor count: 450; parameter count: 215,488,512; alpha: 0.25.

`scripts/materialize_final_release.py` validates all three frozen hashes
before deserializing endpoints, requires exact model keys/shapes/floating
tensors, computes `(0.75 * parent + 0.25 * A4300)` in float32, writes an
atomic no-overwrite artifact, writes provenance, reloads it, and compares every
saved tensor exactly against the FP32 interpolation. Header canonicalization
makes the safetensors byte deterministic: two independent artifacts were
byte-identical and had the SHA above.

`inference.py --release-artifact <artifact>` now loads this public compact
format through the normal control/LoRA compatibility and strict trainable-state
load path; legacy full checkpoint and endpoint interpolation paths remain
unchanged.

## Hero-v1 audit and hero-v2 plan

- Plan: `docs/showcase/final/hero-v2/SHOWCASE_PLAN.md`
- Current accepted inventory: five single-person `mix-025` winners, all at
  control scale `1.0`: psychedelic swordswoman (896x1152), fantasy mage
  (896x1152), comic fashion (1216x832), starry-night painterly (704x1472),
  and dark-fantasy jester (1472x704).
- All five are retained for the expanded 12-image final set. They have clear
  pose-to-image correspondence and collectively cover action fantasy,
  cinematic fantasy, comic fashion, painterly nocturne, and gothic fantasy.
- Proposed additions: realistic female warrior; elegant male warrior/wandering
  knight; moonlit priestess/dreamy floral oracle; stained-glass saint/celestial
  figure; realistic fashion/editorial portrait; gothic masked noble with one
  attendant; painterly mythic companions.
- Proposed final ratio: 10 single-person images and 2 duo images. Duo controls
  are admission-gated on exactly two cleanly separated people and retain
  single-person fallbacks.
- Gap analysis: current winners are fantasy/painterly-heavy with no grounded
  photographic hero, light/celestial or floral register, or intentional duo.
  New concepts correct those gaps without regenerating retained assets.

## Hero-v2 control audition

PASS, pending human selection:

- Review document: `docs/showcase/final/hero-v2/control-audition/AUDITION.md`
- Machine-readable provenance: `docs/showcase/final/hero-v2/control-audition/audition_candidates.json`
- 21 total rendered native-bucket candidate controls: 3 for each of the 7
  new concepts. Every single candidate has authoritative person count 1;
  every duo candidate has authoritative person count 2.
- Contact sheets:
  - `control-audition/realistic-female-warrior_contact_sheet.png`
  - `control-audition/elegant-male-warrior-wandering-knight_contact_sheet.png`
  - `control-audition/moonlit-priestess-dreamy-floral-oracle_contact_sheet.png`
  - `control-audition/stained-glass-saint-celestial-figure_contact_sheet.png`
  - `control-audition/realistic-fashion-editorial-portrait_contact_sheet.png`
  - `control-audition/gothic-masked-noble-with-attendant_contact_sheet.png`
  - `control-audition/painterly-mythic-companions_contact_sheet.png`
- Weak pool: both duo concepts are deliberately cautious. They pass the
  authoritative exactly-two-person and initial separation checks, but their
  sparse/compositional reads require human review at final montage size.

## Files changed this session

- `docs/showcase/final/hero-v2/control-audition/` (controls, 7 contact
  sheets, provenance JSON, and `AUDITION.md`)
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
Read AGENTS/handoff/hero-v2 plan; inspected existing v4/v5 audition controls;
screened authoritative `pose_targets_v3` records by native geometry, person
count, readable head/torso/limbs, crop margin, and (for duos) scale/separation
and low bounding-box overlap; rendered selected controls using the persisted
resize/crop geometry; visually inspected all seven contact sheets. Final
`git diff --check` is required after this handoff update.
```

## Next recommended action

Human selection of one final control per concept. Keep the duo controls
admission-gated until their final composition is confirmed readable at montage
size. Do not generate images or alter frozen evaluation/hero-v1 artifacts as
part of this audition milestone.
