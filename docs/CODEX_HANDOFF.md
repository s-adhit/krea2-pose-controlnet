# Project handoff

## Current objective

Run and inspect the isolated frozen mix-025 native-geometry Krea-2 Turbo control-scale sweep on the GH200. Do not train, access the network from Codex, commit, push, modify canonical `inference.py`, modify training code, or alter any historical benchmark/spec/artifact.

## Frozen control-scale sweep v1

- Immutable spec: `docs/evaluation/control-scale-sweep/mix-025-native-v1.json`; SHA-256 `af82b4c0b0290a852cdd3d7b6b917603e3e9298aa51dfa2ace737d41a28d91a6`.
- Dedicated runner: `scripts/control_scale_sweep.py`; focused tests: `tests/test_control_scale_sweep.py`.
- Candidate is exactly `mix-025`, with the pinned float32 trainable-state interpolation between the recorded parent-4000 and finish-control-a4300 endpoints. The runner verifies resolved endpoint provenance and trainable-state compatibility.
- Geometry is exactly `native_aspect_preserving_cached_latent_bucket`; runtime is Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, pinned Turbo checkpoint path, and the existing `openai/clip-vit-base-patch32` metric.
- Frozen P3-neutral prompting-study conditions, final-val seed, control SHA, and native bucket are: `sculpture_humanart_14000000003803` (simple single), `coco_49731_461706` (dynamic airborne), `real_human_humanart_15000000000521` (inversion), `real_human_humanart_15000000000477` (seated/crouched), and `real_human_humanart_17000000002207` (multi-person).
- Scale order is exactly `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50`: 35 generations. Each pose retains its same frozen seed/prompt across all scales. Scale `0.00` is an internal zero-control reference only; it does not replace the later full Turbo baseline.
- The local zero-scale sampler otherwise uses the locked Turbo schedule/runtime and final-val model/scoring mechanics. It exists only because the generic positive-control helper intentionally rejects scale 0.
- Fail-closed checks cover immutable spec/source-prompt/final-val/candidate/control-hash/native-bucket/runtime/CLIP drift and incomplete or inconsistent output roots. No RGB source fallback exists.
- Expected artifacts: `control_scale_provenance.json`, `pck_clip_results.json`, `metrics_by_control_scale.json`, `metrics_by_pose_class.json`, `evaluation_summary.json`, `compact_summary.json`, `control_scale_contact_sheet.png`, and `per_pose_scale_grids/*.png`.

## Exact GH200 commands

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1
```

```bash
uv run python scripts/control_scale_sweep.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1
uv run python scripts/control_scale_sweep.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1
uv run python scripts/control_scale_sweep.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/control_scale_sweep.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1
uv run python scripts/control_scale_sweep.py summary --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1
```

## Files changed this session

- `docs/evaluation/control-scale-sweep/mix-025-native-v1.json`
- `scripts/control_scale_sweep.py`
- `tests/test_control_scale_sweep.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/control_scale_sweep.py tests/test_control_scale_sweep.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_control_scale_sweep tests.test_prompting_guide_study tests.test_multilingual_prompt_smoke tests.test_chinese_prompt_smoke tests.test_final_val_turbo_benchmark -v
# 39 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/control_scale_sweep.py --help
git diff --check
```

No generation, scoring, network access, commit, or push was performed in Codex.

## Next action

Run the five commands above from the GH200 host in order, inspect the seven-column per-pose grids and aggregate scale metrics, then use this result as an internal control-strength selection aid only.
