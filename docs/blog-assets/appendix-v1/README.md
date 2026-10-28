# Appendix v1

The full reproducibility archive contains eight validated controls × three
short, geometry-neutral appearance families: 24 successful native-geometry
generations. The skeleton is the sole body-geometry source; prompts specify
only visual treatment. The frozen runtime is Krea-2 Turbo, 8 steps, CFG 0,
mu 1.15, control scale 1.0, no Style-LoRA, and native/aspect-preserving
geometry selected with `choose_bucket(source_size, REFERENCE_KREA_BUCKETS)`.

`manifest.json` is the source of reproducibility facts. `contact_sheet.png`
remains the complete archive sheet, with one row per control and columns for
`control` plus all three style families. It and the full 24-image generation
set are retained intact as supplementary and limitations material.

`curated_contact_sheet.png` is the blog appendix figure. It uses six controls
(`single_01_female_warrior`, `single_02_wandering_knight`,
`single_03_floral_oracle`, `single_04_stained_glass_saint`,
`single_07_crouched`, and `multi_01_gothic_duo`) × two style families
(`cinematic_fantasy_realism` and `painterly_storybook_fantasy`). Its columns
are `control`, `cinematic_fantasy_realism`, and
`painterly_storybook_fantasy`; images are letterboxed, never cropped.

The omitted examples remain available in the full archive as supplementary
and limitations material. `single_08_overhead_arms` and dense multi-person
cases are not part of the curated figure because their pose adherence is
weaker. Frozen native-vs-dynamic and hard-pose comparison assets are separate
from this appendix selection and remain unchanged.
