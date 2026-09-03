# Project handoff

## Current objective

Run and inspect the frozen, isolated Krea-2 Turbo `mix-025` Style-LoRA strength sweep. Do not train, access the network from Codex, commit, push, alter canonical `inference.py`, alter training code, alter historical v1/v2 artifacts, or begin another composition/grid experiment.

## Frozen strength-sweep-v1 contract

- Immutable spec: `docs/evaluation/style-lora-composition/style_lora_strength_sweep_v1.jsonl`; SHA-256 `4dd68fb2773c31122f5289cc52f14d8f5334558d3a364e9d4b83e8721f2e5fdb`. The loader validates it fail-closed and verifies its stems, semantic base prompts, triggers, and Style-LoRA hashes match frozen v2.
- Candidate: `mix-025`, FP32 `(0.75 * parent-4000) + (0.25 * finish-control-a4300)` over only `state['model']` trainable tensors. Pinned endpoint hashes are revalidated at every stage.
- Runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, no resolution-dependent mu shift, native/aspect-preserving cached-latent geometry, pose/control scale `1.0`, and frozen final-val sampling seed per pose. Source RGB is neither sampled nor used as fallback.
- Poses: `simple_single` / `sculpture_humanart_14000000003803`; `dynamic_airborne` / `coco_49731_461706`; `inversion` / `real_human_humanart_15000000000521`; `multi_person` / `real_human_humanart_17000000001263`.
- Styled cells: darkbrush, rainywindow, retroanime, realism at exactly `[0.25, 0.50, 0.75, 1.00]` in that order: 64 styled generations. Generate one pose-only baseline per pose (4 more), for 68 total. The sweep does not permit `--style-strength`; per-cell values are spec-pinned.
- Prompts: darkbrush appends `, monochrome ink wash style`; rainywindow appends `, rainy window style`; retroanime appends `, Purple retro anime style`; realism and pose-only have no trigger. CLIP uses this exact effective prompt. Style fidelity is qualitative only.
- Style adapters retain the pinned hashes and strict 528-FP32-tensor / 264-pair / rank-32 mapping audit. They are scoped temporary hooks and never merge into Pose-LoRA state.

## Isolation and output contract

- New output root only: `/lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1`.
- Immutable provenance records the spec, candidate interpolation, Turbo/control contract, prompts, style hashes, seeds, control hashes, geometry, exact styles, strengths, and counts. Drift or v1/v2 output-root identity collision fails closed.
- Generation rejects a partial matrix. Reporting requires all 64 styled cells plus exactly four pose-only baselines.
- Outputs: `metrics_by_style_strength.json` (primary, with PCK@0.05/0.10/0.20, coverage/person counts, CLIP, and pose-only deltas); `metrics_by_style.json` and `metrics_by_strength.json` (secondary diagnostics); `pose_retention_vs_strength.json`; grids; contact sheet; summary; provenance; compact summary.

## Files changed this session

- `docs/evaluation/style-lora-composition/style_lora_strength_sweep_v1.jsonl`
- `scripts/style_lora_composition.py`
- `tests/test_style_lora_composition.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/style_lora_composition.py tests/test_style_lora_composition.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_style_lora_composition -v
# 11 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/style_lora_composition.py --help
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -c "... load_rows('strength-sweep-v1') ..."
# frozen SHA verified; 68 total / 64 styled generations
git diff --check
```

Tests cover frozen style/strength lists and ordering; v2 semantic/control parity; exact triggers/effective prompts at every strength; per-strength metadata invariance except Style-LoRA scale; fixed pose scale; scoped hook non-leakage; seed/control identity; full 64-cell completeness; one pose-only baseline per pose; immutable provenance drift and v1/v2 collision failure; and baseline-delta aggregation.

## Exact GH200 commands

```bash
uv run python scripts/style_lora_composition.py audit --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1
uv run python scripts/style_lora_composition.py preflight --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1
uv run python scripts/style_lora_composition.py generate --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1
uv run python scripts/style_lora_composition.py score --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/style_lora_composition.py report --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1
uv run python scripts/style_lora_composition.py summary --experiment strength-sweep-v1 --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/strength-sweep-v1
```

## Next action

Run and inspect the Style-LoRA strength sweep; identify the lowest qualitatively clear style strength with acceptable pose retention. Do not hard-code a recommendation before review.
