# Project handoff

## Current objective

Materialize the 48-image qualitative appendix on the CUDA-visible GH200 shell
before archival. The appendix now uses the frozen native/aspect-preserving
bucket policy for every copied control; do not alter release identity,
checkpoints, prompts, seeds, renderer, metrics, or Turbo runtime defaults.
The audit shell has no visible CUDA.

## Frozen execution contract

- Release: `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors`, SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Base sampler: Krea-2 Turbo at `/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors`; Raw provenance remains `/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`.
- Runtime: 8 steps, CFG 0, `mu=1.15`, control scale 1.0, no Style-LoRA, deterministic per-row seed, native/aspect-preserving geometry.
- Prompting: consult `prompting.md`; new prompts never narrate pose/body geometry. The control is authoritative for count, framing, and anatomy.
- Geometry resolver: `choose_bucket(source_size, REFERENCE_KREA_BUCKETS)` from
  `pose_controlnet.paired_preprocessing`, selecting the nearest log-aspect
  frozen native bucket (not dynamic-768 or a forced square). Generation passes
  that resolved width and height explicitly to `inference.py`.

## Prepared / green

- `scripts/prepare_qualitative_appendix.py` validates the frozen release SHA, every source control/output, output-path uniqueness, and a geometry-word prompt linter before any model load. `--generate-appendix` requires CUDA and reuses one inference runtime for the 48 planned samples. It stops on a generation exception; no silent retries occur.
- `docs/blog-assets/appendix-v1/manifest.json` contains exactly 48 deterministic rows: 12 controls (8 single, 4 duo/multi) × four appearance families. It now records source dimensions plus the resolved generation `width`/`height`; all are positive and divisible by 16.
- Twenty existing appendix images are valid at their resolved native buckets and will be retained. Four old `single_06_seated` 640×640 appendix outputs are classified as stale and will be replaced only during the explicit CUDA generation, together with the 24 missing rows. Frozen blog examples are not touched.
- Reused (not regenerated) eight frozen evaluation outputs and their controls:
  - `docs/blog-assets/qualitative/native-vs-dynamic/`: inversion portrait and multi-person stage example, with native/dynamic triptych sheet.
  - `docs/blog-assets/qualitative/hard-pose-single-multi/`: strong-foreshortening and inversion singles; two-person interaction and overlapping-pair multis, with labelled sheet.
- `docs/blog-assets/SHA256SUMS.txt` covers all currently materialized target assets and will be rewritten after appendix completion.
- Visual inspection passed for both targeted comparison sheets; images are letterboxed (never cropped) and labels identify the comparison classes.

## Exact checks run

- PASS: `uv run python -m unittest tests.test_prepare_qualitative_appendix tests.test_inference tests.test_paired_preprocessing` — 23 tests, including non-divisible square, portrait, landscape, already-valid native bucket, all-48 manifest resolution, and stale appendix classification.
- PASS: `python scripts/prepare_qualitative_appendix.py` — 48 manifest rows validate without model load; 20 success, 28 `pending_cuda`; eight frozen evaluation outputs reused.
- PASS: `git diff --check`.
- INFO: `uv run python -c 'import torch; print(torch.cuda.is_available())'` reported `False`; no appendix generation was launched in this shell.

## Files changed this session

- `scripts/prepare_qualitative_appendix.py`
- `tests/test_prepare_qualitative_appendix.py`
- `docs/blog-assets/appendix-v1/` (manifest, README, 12 copied controls; samples pending)
- `docs/blog-assets/qualitative/native-vs-dynamic/` (2 frozen triptychs and manifest)
- `docs/blog-assets/qualitative/hard-pose-single-multi/` (4 frozen pairs and manifest)
- `docs/blog-assets/SHA256SUMS.txt`
- `docs/blog-assets/BLOG_ASSET_INDEX.md`
- `docs/CODEX_HANDOFF.md`

## Exact next action

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/prepare_qualitative_appendix.py --generate-appendix
```

This keeps the 20 valid appendix images, replaces the four stale 640×640
appendix images at 1024×1024, and generates the other 24 pending images. On
success, inspect `docs/blog-assets/appendix-v1/contact_sheet.png`, rerun
`git diff --check`, and do not commit or push unless explicitly authorized.
