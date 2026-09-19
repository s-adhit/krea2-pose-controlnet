# Project handoff

## Current objective

Materialize the approved five-concept `hero-v2` retry-a batch on the writable
GH200 host, then conduct human visual review. Do not modify `canonical-v1`,
the two accepted canonical-v1 keepers, hero-v1, or evaluation artifacts.

## Hero-v2 retry-a status

- Manifest: `docs/showcase/final/hero-v2/retries/retry-a/retry_a.json`
- Readable plan: `docs/showcase/final/hero-v2/retries/retry-a/RETRY_A.md`
- Runner: `scripts/generate_hero_v2_retry.py`; it reuses canonical release
  validation/no-overwrite helpers and `inference.py` generation/runtime.
- NFS output directory: `/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/retry-a/`
- Repo presentation directory:
  `docs/showcase/final/hero-v2/generations/retry-a/`
- Retry set: realistic female warrior (replacement native 1216x832 control,
  seed 7194308501, scale 1.0); moonlit priestess (7194308502, 1.0);
  stained-glass saint (7194308503, 1.0); fashion/editorial portrait
  (7194308504, 1.0); painterly mythic companions (7194308505, 1.25).
  The latter four retain their frozen canonical controls.
- `uv run python -m py_compile scripts/generate_hero_v2_retry.py`: PASS.
- `uv run python scripts/generate_hero_v2_retry.py preflight`: PASS. It
  validated all five native control geometries, exact manifest contract, and
  release artifact SHA. No generation was attempted in this sandbox.
- This Codex sandbox's `/lambda/nfs` is read-only. Run the command below from
  the writable GH200/Lambda host; the runner refuses to overwrite differing
  NFS or presentation files.

## Hero-v2 canonical generation status

- Runner: `scripts/generate_hero_v2_canonical.py`; it reuses `inference.py`
  request/runtime/generation functions (no sampler or model-loader copy).
- Required output directory: `/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/canonical-v1/`
- Repo presentation target: `docs/showcase/final/hero-v2/generations/canonical-v1/`
- The runner validates the frozen selection, each control's native dimensions,
  frozen release runtime, and release artifact SHA before it loads a model. It
  generates only scale-1.0 candidates and writes augmented adjacent canonical
  sidecars with concept/stem/release fields, then packages copies and a
  control-plus-output contact sheet.
- `uv run python scripts/generate_hero_v2_canonical.py preflight`: PASS (all
  seven frozen controls and release SHA validated).
- `uv run python scripts/generate_hero_v2_canonical.py generate`: BLOCKED
  before model load or output creation. This Codex sandbox mounts `/lambda/nfs`
  read-only (`OSError: [Errno 30] Read-only file system` creating the required
  output directory). No concepts were generated and no contact sheet exists.
- Run the same command from the writable GH200/Lambda host shell. It will
  write `/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/canonical-v1/` and
  `docs/showcase/final/hero-v2/generations/canonical-v1/canonical-v1_contact_sheet.png`.

## Next recommended action

From the writable GH200 host, run:

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/generate_hero_v2_retry.py preflight
uv run python scripts/generate_hero_v2_retry.py generate
```

Then inspect `docs/showcase/final/hero-v2/generations/retry-a/retry-a_contact_sheet.png`
and select only through human review. Do not commit or launch production training.

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

## Hero-v2 frozen selection and generation plan

PASS: the seven new controls are frozen in
`docs/showcase/final/hero-v2/final-selection/FINAL_SELECTION.md` and the
machine-readable `final_selection.json`. The plan fixes `mix-025`, turbo mode,
native aspect-preserving geometry, control scale `1.0`, and one seed per
concept. The two duo concepts allow a `1.25` retry only for weak adherence.

Frozen selected stems:

- realistic female warrior: `real_human_humanart_15000000000016`
- elegant male warrior / wandering knight: `painting_humanart_9000000000455`
- moonlit priestess / dreamy floral oracle: `painting_humanart_9000000000724`
- stained-glass saint / celestial figure: `sculpture_humanart_14000000004082`
- realistic fashion/editorial portrait: `painting_humanart_9000000001986`
- gothic masked noble with attendant: `real_human_humanart_15000000002158`
- painterly mythic companions: `painting_humanart_9000000000976`

The preceding control-only audition remains the provenance source:

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

- `docs/showcase/final/hero-v2/final-selection/FINAL_SELECTION.md`
- `docs/showcase/final/hero-v2/final-selection/final_selection.json`
- `scripts/generate_hero_v2_canonical.py`
- `docs/showcase/final/hero-v2/retries/retry-a/retry_a.json`
- `docs/showcase/final/hero-v2/retries/retry-a/RETRY_A.md`
- `scripts/generate_hero_v2_retry.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
Read AGENTS/handoff/hero-v2 plan/audition and `prompting.md`; verified every
selected stem, native dimensions, person count, and rendered-control path
against `audition_candidates.json`. Prompts are geometry-neutral, retain the
authoritative subject count, and avoid framing or joint-by-joint instructions.
Final `git diff --check` is required after this handoff update.
```
