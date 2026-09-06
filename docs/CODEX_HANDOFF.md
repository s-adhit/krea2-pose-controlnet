# Project handoff

## Current objective

Run the isolated frozen Krea-2 Turbo baseline versus Pose-Control comparison on the GH200. Do not train, access the network from Codex, commit, push, modify canonical `inference.py`, modify training code, or modify historical benchmark/spec/artifact namespaces.

## Frozen Turbo baseline final-val comparison v1

- Immutable spec: `docs/evaluation/turbo-baseline/turbo-baseline-final-val-v1.json`; SHA-256 `f0775c8aac404a1a4a3303ee8272d49d217d398c8379d7559d84c59288292bcc`.
- Dedicated runner: `scripts/turbo_baseline_benchmark.py`; focused tests: `tests/test_turbo_baseline_benchmark.py`.
- Exact candidate order is `turbo-base`, `parent-4000`, `mix-025`, `finish-control-a4300`. It is exactly 48 frozen final-val conditions × 4 candidates = 192 generations.
- Runtime is exactly Krea-2 Turbo, native/aspect-preserving cached-latent geometry, 8 steps, CFG 0, `mu=1.15`. Pose-Control candidates use control scale `1.0`.
- `turbo-base` builds the official Turbo checkpoint into the native `SingleStreamDiT` only. It does not construct `ControlInputLayer`, inject LoRA, load a trainable Pose-LoRA/control-adapter tensor, or use a zeroed control latent. Its SHA-256 is captured during preflight in `turbo_baseline_provenance.json`; every later stage recomputes it and refuses any drift.
- `parent-4000` and `finish-control-a4300` retain their pinned checkpoint SHA-256 values. `mix-025` is pinned to the FP32 `(1 - alpha) * parent-4000 + alpha * finish-control-a4300` interpolation at `alpha=0.25`, over trainable control/LoRA tensors only.
- The runner verifies final-val spec/prompt/sidecar hashes, cached sample identities and seeds, source caption mapping, physical control hashes, native geometry, candidate hashes, Turbo identity, output dimensions, historical artifact hashes, and all-or-nothing output roots. It does not use RGB fallback.
- Outputs: `turbo_baseline_provenance.json`, `generation_results.json`, `pck_clip_results.json`, `metrics_by_candidate.json`, `evaluation_summary.json`, `compact_summary.json`, `turbo_baseline_contact_sheet.png`, plus one contact sheet per candidate.
- Scoring preserves every image-level PCK/CLIP result and aggregates PCK@0.05/0.10/0.20, CLIP, matched people, detection coverage, generated people, unmatched reference/predicted people, plus available single-/multi-person and source-domain breakdowns.

## Exact GH200 commands

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1
```

```bash
uv run python scripts/turbo_baseline_benchmark.py preflight --output-root /lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1
uv run python scripts/turbo_baseline_benchmark.py generate --output-root /lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1
uv run python scripts/turbo_baseline_benchmark.py score --output-root /lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/turbo_baseline_benchmark.py report --output-root /lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1
uv run python scripts/turbo_baseline_benchmark.py summary --output-root /lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1
```

`preflight` is mandatory: it is the stage that pins the exact local 26 GB Turbo checkpoint hash before any generation. Later stages intentionally refuse to run without matching preflight provenance.

## Files changed this session

- `docs/evaluation/turbo-baseline/turbo-baseline-final-val-v1.json`
- `scripts/turbo_baseline_benchmark.py`
- `tests/test_turbo_baseline_benchmark.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/turbo_baseline_benchmark.py tests/test_turbo_baseline_benchmark.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_turbo_baseline_benchmark tests.test_final_val_turbo_benchmark tests.test_native_vs_dynamic768 -v
# 26 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/turbo_baseline_benchmark.py --help
git diff --check
```

No generation, scoring, network access, commit, or push was performed in Codex.

## Next action

Run the five GH200 commands above in order. Review `compact_summary.json`, `metrics_by_candidate.json`, and the baseline/four-candidate contact sheets only after the staged run completes.
