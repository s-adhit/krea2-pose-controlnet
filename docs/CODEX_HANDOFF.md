# Project handoff

## Current objective

Publication-ready blog assets have been generated from frozen training,
evaluation, infrastructure, and release records. They await human review,
then blog prose integration. Do not alter frozen evaluation artifacts, NFS
checkpoints, release identity, or the existing hero-v2 winners.

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

## Verification this session

PASS:

```bash
MPLCONFIGDIR=/tmp/krea2-blog-mpl uv run python scripts/generate_blog_assets.py
# visual inspection of all eight PNG figures
# indexed PNG, SVG, and table paths exist
git diff --check
```

## Files changed this session

- `scripts/generate_blog_assets.py`
- `docs/blog-assets/` (generated figures, tables, and index)
- `docs/CODEX_HANDOFF.md`

Pre-existing/unrelated untracked file: `scripts/package_hero_v2_final.py`.
Do not commit or push without explicit authorization.

## Remaining visual/content decisions

- Human review of chart typography, dark-theme embedding, and final blog
  captions is the remaining visual decision.
- The figures deliberately do not claim statistical significance or convert
  descriptive subset results into generalization claims.

## Next recommended action

Human review of `docs/blog-assets/`, then write blog prose using the asset
index and frozen evidence records. Preserve the approved architecture asset
unchanged.
