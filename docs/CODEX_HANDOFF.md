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

The appendix is now an 8-control × 3-style, 24-generation plan. Retained
controls are `single_01_female_warrior`, `single_02_wandering_knight`,
`single_03_floral_oracle`, `single_04_stained_glass_saint`,
`single_07_crouched`, `single_08_overhead_arms`, `multi_01_gothic_duo`, and
`multi_04_dense_group`. It has six single-person and two multi-person rows.
The style families are `cinematic_fantasy_realism`,
`painterly_storybook_fantasy`, and `classical_oil_painting`.

Preparation validates release provenance, source/copy control rasters, prompt
neutrality, native bucket dimensions, unique outputs, and full relevant
sidecar runtime/control/release metadata without loading the model. A retained
output is reusable only if that metadata and its prompt match the current row.
The current plan has 0 valid reuses; the 16 old retained cinematic/painterly
outputs are stale due to changed prompts, and the 8 classical-oil outputs are
absent. All 24 are therefore pending CUDA. The stale `contact_sheet.png` was
removed; successful materialization will make an eight-row sheet with columns
`control | cinematic_fantasy_realism | painterly_storybook_fantasy |
classical_oil_painting`, preserving aspect ratio by letterboxing.

Appendix-only cleanup removed the four dropped control rasters, 64 dropped
control/style PNG and JSON artifacts, and the old contact sheet. Retained
control rasters, source provenance, deterministic seeds, source dimensions,
resolved native buckets, frozen runtime, and comparison assets were preserved.

## Verified this session

- PASS: `uv run python -m unittest tests.test_prepare_qualitative_appendix tests.test_inference tests.test_paired_preprocessing` — 29 tests.
- PASS: `python scripts/prepare_qualitative_appendix.py` — 24 planned, 0 valid/reused, 24 pending/stale (16 stale), and 8 targeted frozen evaluation outputs reused unchanged.
- INFO: `uv run python -c 'import torch; print(torch.cuda.is_available())'` reported `False`; no generation was launched in this Codex shell.

## Files changed this session

- `scripts/prepare_qualitative_appendix.py`
- `tests/test_prepare_qualitative_appendix.py`
- `docs/blog-assets/appendix-v1/README.md`
- `docs/blog-assets/appendix-v1/manifest.json`
- `docs/blog-assets/appendix-v1/` obsolete files removed as described above
- `docs/blog-assets/BLOG_ASSET_INDEX.md`
- `docs/blog-assets/SHA256SUMS.txt`
- `docs/CODEX_HANDOFF.md`

## Exact next action

Only from a CUDA-visible GH200 shell:

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/prepare_qualitative_appendix.py --generate-appendix
```

Then rerun `python scripts/prepare_qualitative_appendix.py`, inspect the
four-column image layout, run the targeted test command above, and run
`git diff --check`. Do not commit or push without explicit authorization.
