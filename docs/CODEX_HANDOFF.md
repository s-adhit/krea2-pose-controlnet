# Project handoff

## Current objective

Publication-ready blog assets have been generated from frozen training,
evaluation, infrastructure, release records, and a representative paired
training-data sample. The Data and conditioning montages were revised to a
full-source justified editorial layout and await human review, then blog prose
integration.
Do not alter frozen evaluation artifacts, NFS checkpoints, release identity,
or the existing hero-v2 winners.

The existing architecture figure is user-approved and intentionally unchanged.
It was neither recreated nor redesigned during this asset pass.

## Frozen project facts in force

- Release `mix-025` / `krea2-pose-control-lora-v1` is float32 interpolation of
  trainable `state['model']` only: `0.75 * parent-4000 + 0.25 * A4300`.
  Release SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Endpoint: 215,488,512 float32 trainable parameters / 450 tensors;
  expanded control projection plus rank-64 LoRA; backbone frozen. Objective:
  flow MSE plus `.04` normalized-coordinate Huber at shifted timesteps
  `[.10,.20]`.
- Normal-host evidence: ARM64 GH200; Python 3.10.12, PyTorch 2.7.0/CUDA 12.8,
  cuDNN 9.8, Triton 3.3.0. The Codex audit sandbox does not expose CUDA; this
  does not contradict the host record.
- Frozen evaluation methods/results are authoritative in
  `docs/blog-evidence/EVALUATION_METHODS.md`, `evaluation_methods.json`, and
  `EVALUATION_RESULTS.md`. PCK is globally pooled eligible joints; CLIP is
  mean image/prompt cosine similarity; native/dynamic is not a pure
  resolution-only comparison.

## Generated blog assets

- Generator: `scripts/generate_blog_assets.py` (Matplotlib only; no seaborn).
  It exports purple-obsidian themed high-resolution PNG and SVG figures.
- Figure pairs in `docs/blog-assets/figures/`:
  `training_objective_overview`, `training_lineage`, `final_val_pck`,
  `final_val_clip`, `native_vs_dynamic`, `hard_pose_single_multi`,
  `prompt_injection_effect`, and `infra_summary`.
- Tables in `docs/blog-assets/tables/`: `training_recipe.md`,
  `training_stages.md`, `evaluation_method_summary.md`, `final_val_results.md`,
  `ablation_results.md`, and `release_identity.md`.
- `docs/blog-assets/BLOG_ASSET_INDEX.md` gives intended section, paths,
  frozen sources, captions/caveats, and status for every logical asset.
- Exact values remain in the frozen result record and the generator constants;
  figure labels are presentation rounding only. The CLIP charts retain an
  absolute `0–0.40` axis so small differences are not visually exaggerated.
- `scripts/generate_dataset_montage.py` renders an exact paired 24-sample
  RGB/control justified editorial montage from the read-only PoseBridge
  snapshot. It uses uncropped, aspect-ratio-preserving source images and its
  selection is checked against both frozen train manifests. It outputs
  `figures/dataset_rgb_montage.png`, `figures/dataset_condition_montage.png`,
  and `dataset_montage_manifest.json`. The manifest records order, paths,
  source/display geometry, full-image bounds, shared cell positions, source
  domain, hashes, and available authoritative person counts. Controls are
  existing source files, not regenerated or recolored.

## Verification this session

PASS:

```bash
MPLCONFIGDIR=/tmp/krea2-blog-mpl uv run python scripts/generate_blog_assets.py
# visual inspection of all eight PNG figures
# indexed PNG, SVG, and table paths exist
git diff --check
```

PASS (dataset montage session):

```bash
uv run python scripts/generate_dataset_montage.py
# visual inspection of both dataset montage PNGs: full-source justified layout
# and shared paired positions confirmed
# inline Python provenance audit of both frozen train manifests, source/control
# stems, dimensions, and all recorded SHA-256 hashes
# PASS: hashes, stems, geometry, and dual frozen-train provenance verified for
# 24 paired samples
git diff --check
```

## Files changed this session

- `scripts/generate_dataset_montage.py`
- `docs/blog-assets/figures/dataset_rgb_montage.png`
- `docs/blog-assets/figures/dataset_condition_montage.png`
- `docs/blog-assets/dataset_montage_manifest.json`
- `docs/blog-assets/BLOG_ASSET_INDEX.md`
- `docs/CODEX_HANDOFF.md`

Pre-existing/unrelated untracked file: `scripts/package_hero_v2_final.py`.
Do not commit or push without explicit authorization.

## Remaining visual/content decisions

- Human review of chart typography, dark-theme embedding, and final blog
  captions is the remaining visual decision.
- The figures deliberately do not claim statistical significance or convert
  descriptive subset results into generalization claims.

## Next recommended action

Human review of `docs/blog-assets/`, including the paired full-source dataset
montages, then write blog prose using the asset index and frozen evidence records.
Preserve the approved architecture asset unchanged.
