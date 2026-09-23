# Project handoff

## Current objective

Before GH200/NFS archival, materialize the prepared 48-image qualitative
appendix and retain the small targeted blog examples. Do not train, alter the
release identity/checkpoints/metrics/renderer, or change frozen runtime
defaults. The audit shell used for preparation has no visible CUDA; generation
must resume on the normal CUDA-visible GH200 shell.

## Frozen execution contract

- Release: `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors`, SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Base sampler: Krea-2 Turbo at `/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors`; Raw provenance remains `/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`.
- Runtime: 8 steps, CFG 0, `mu=1.15`, control scale 1.0, no Style-LoRA, deterministic per-row seed, native/aspect-preserving geometry.
- Prompting: consult `prompting.md`; new prompts never narrate pose/body geometry. The control is authoritative for count, framing, and anatomy.

## Prepared / green

- Added `scripts/prepare_qualitative_appendix.py`, a two-phase asset tool. It validates the frozen release SHA, every source control/output, output-path uniqueness, and a geometry-word prompt linter before any model load. `--prepare-only` is implicit; `--generate-appendix` requires CUDA and reuses one inference runtime for the 48 planned samples. It stops on a generation exception; no silent retries occur.
- `docs/blog-assets/appendix-v1/manifest.json` contains exactly 48 pending deterministic rows: 12 controls (8 single, 4 duo/multi) × cinematic fantasy realism, painterly storybook fantasy, polished modern anime, and editorial high-fashion photography. Copied controls and `README.md` are present; `generations/` and `contact_sheet.png` await GH200 sampling.
- Reused (not regenerated) eight frozen evaluation outputs and their controls:
  - `docs/blog-assets/qualitative/native-vs-dynamic/`: inversion portrait and multi-person stage example, with native/dynamic triptych sheet.
  - `docs/blog-assets/qualitative/hard-pose-single-multi/`: strong-foreshortening and inversion singles; two-person interaction and overlapping-pair multis, with labelled sheet.
- `docs/blog-assets/SHA256SUMS.txt` covers all currently materialized target assets and will be rewritten after appendix completion.
- Visual inspection passed for both targeted comparison sheets; images are letterboxed (never cropped) and labels identify the comparison classes.

## Exact checks run

- PASS: `python scripts/prepare_qualitative_appendix.py` — release SHA and all source files validate; 0 / 48 appendix samples generated (correctly `pending_cuda`); 8 frozen evaluation outputs reused.
- PASS: Pillow verify over all materialized qualitative PNGs.
- PASS: manifest audit: 48 rows, 48 unique planned output paths, four finalized prompt families.
- INFO: audit shell `torch.cuda.is_available()` is false and `nvidia-smi` cannot communicate with a driver; this does not invalidate host GH200 verification.

## Files changed this session

- `scripts/prepare_qualitative_appendix.py`
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

On success, rerun `git diff --check` and inspect `docs/blog-assets/appendix-v1/contact_sheet.png`. The script refreshes manifests and checksums. Do not commit or push unless explicitly authorized.
