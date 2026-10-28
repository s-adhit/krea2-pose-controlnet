# Current handoff

## Project state

Training, evaluation, release selection, and ComfyUI integration are complete.
The frozen release is `krea2-pose-control-lora-v1`, materialized artifact
`krea2-pose-control-mix025.safetensors`, SHA-256
`6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
Krea-2 Raw is training provenance; frozen inference is Krea-2 Turbo at 8
steps, CFG 0, `mu=1.15`, control scale 1.0, native/aspect-preserving geometry,
and no Style-LoRA. Do not regenerate frozen native-vs-dynamic or hard-pose
comparison assets, change release identity, or train.

## Appendix v1

The complete appendix archive is successful and intact: eight controls ×
three style families = 24 generation PNGs, with their sidecars and immutable
`manifest.json`. `contact_sheet.png` remains the full archive sheet. Its
controls are `single_01_female_warrior`, `single_02_wandering_knight`,
`single_03_floral_oracle`, `single_04_stained_glass_saint`,
`single_07_crouched`, `single_08_overhead_arms`, `multi_01_gothic_duo`, and
`multi_04_dense_group`; its styles are `cinematic_fantasy_realism`,
`painterly_storybook_fantasy`, and `classical_oil_painting`.

This session created the presentation-only
`curated_contact_sheet.png`: six rows (`single_01_female_warrior`,
`single_02_wandering_knight`, `single_03_floral_oracle`,
`single_04_stained_glass_saint`, `single_07_crouched`,
`multi_01_gothic_duo`) × two styles (`cinematic_fantasy_realism`,
`painterly_storybook_fantasy`). It must use only the existing successful
PNGs, preserve aspect ratio with letterboxing, and never run inference or
modify generation results. Omitted examples remain supplementary/limitations
material; `single_08_overhead_arms` and dense multi-person cases are omitted
from the curated figure because pose adherence is weaker.

## Current session verification

- PASS: the manifest has 24 successful records; all 12 selected outputs and
  all six selected controls opened successfully with Pillow verification/load.
- PASS: `curated_contact_sheet.png` exists, opens, and is 1246 × 2260 px. It
  was composed through the existing `build_sheet` compositor, which uses
  `thumbnail` and centered placement, so it letterboxes rather than crops.
- PASS: the curated sheet checksum is recorded and matched in
  `docs/blog-assets/SHA256SUMS.txt`; `sha256sum --check --status` passed.
- PASS: the full `contact_sheet.png` remains present and all 24 generation
  PNGs remain present. No inference or generation function was invoked.
- PASS: `git diff --check`.

## Files changed this session

- `docs/blog-assets/appendix-v1/curated_contact_sheet.png`
- `docs/blog-assets/appendix-v1/README.md`
- `docs/blog-assets/BLOG_ASSET_INDEX.md`
- `docs/blog-assets/SHA256SUMS.txt`
- `docs/CODEX_HANDOFF.md`

The appendix manifest and generation PNG/JSON worktree changes that already
existed at session start were not modified by this curation task. Frozen
native-vs-dynamic and hard-pose comparison assets remain untouched.

## Exact next action

Review and stage the intended documentation, checksum, and curated figure
changes. Do not commit or push without explicit authorization.
