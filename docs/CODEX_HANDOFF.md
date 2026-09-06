# Project handoff

## Current objective

Run the isolated frozen native-cache versus dynamic-768 Krea-2 Turbo release ablation on the GH200. Do not train, access the network from Codex, commit, push, modify canonical `inference.py`, modify training code, or alter historical benchmark/spec/artifact namespaces.

## Frozen native-vs-dynamic-768 ablation v1

- Immutable spec: `docs/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1.json`; SHA-256 `fe2d109e1e08c05007e38f605306ce1f9092794d0b7e0017a553e1fd3d6906cf`.
- Dedicated runner: `scripts/native_vs_dynamic768.py`; focused tests: `tests/test_native_vs_dynamic768.py`.
- Candidate is exactly `mix-025`, the pinned float32 trainable-state interpolation between parent-4000 and finish-control-a4300. Control scale is exactly `1.0`.
- Runtime is exactly Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, pinned Turbo checkpoint, and `openai/clip-vit-base-patch32` scoring.
- Ordered geometry modes are exactly `native_aspect_preserving_cached_latent_bucket`, then `dynamic_768_bucket`: 5 conditions × 2 = exactly 10 generations. Native consumes its cached latent; dynamic applies the shared 768 bucket policy to the authoritative control raster and VAE-encodes only that raster. No RGB fallback exists.
- The frozen final-val conditions are: `sculpture_humanart_14000000003803` (near-square, 1024×1024 -> 768×768), `real_human_humanart_15000000000521` (portrait, 832×1216 -> 640×960), `real_human_humanart_15000000000477` (landscape, 1472×704 -> 1024×576), `sculpture_humanart_14000000000288` (extreme portrait, 768×1344 -> 576×1024), and `real_human_humanart_17000000002207` (multi-person, 1216×832 -> 960×640).
- Prompt text is frozen from `final_val_benchmark_48.jsonl`; each condition retains the same exact prompt and seed in both geometry modes. The spec binds final-val seed, control SHA, source size, native bucket, and dynamic bucket per condition.
- The runner rejects spec/hash/runtime/candidate/control/control-scale/prompt/final-val/historical-artifact drift, conflicting output provenance, corrupt/wrong-dimension output, and incomplete output roots. It records output dimensions/buckets, PCK@0.05/0.10/0.20, CLIP cosine, matched people, detection coverage, and existing-geometry crop/framing comparison.
- Expected output files: `native_dynamic_provenance.json`, `pck_clip_results.json`, `metrics_by_geometry.json`, `metrics_by_condition.json`, `evaluation_summary.json`, `compact_summary.json`, `native_vs_dynamic768_contact_sheet.png`, and `per_condition_grids/*.png` (columns: pose control | native | dynamic-768).

## Exact GH200 commands

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1
```

```bash
uv run python scripts/native_vs_dynamic768.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1
uv run python scripts/native_vs_dynamic768.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1
uv run python scripts/native_vs_dynamic768.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/native_vs_dynamic768.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1
uv run python scripts/native_vs_dynamic768.py summary --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1
```

## Files changed this session

- `docs/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1.json`
- `scripts/native_vs_dynamic768.py`
- `tests/test_native_vs_dynamic768.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/native_vs_dynamic768.py tests/test_native_vs_dynamic768.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_native_vs_dynamic768 tests.test_control_scale_sweep tests.test_final_val_turbo_benchmark -v
# 26 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/native_vs_dynamic768.py --help
git diff --check
```

No generation, scoring, network access, commit, or push was performed in Codex.

## Next action

Run the five commands above from the GH200 host in order. Inspect the three-column per-condition grids and the geometry/condition aggregates; use this only as the frozen release geometry ablation.
