# Project handoff

## Current objective

The frozen 48-image final-val Turbo evaluator now supports exactly three
in-memory trainable-state interpolation candidates: `mix-025`, `mix-050`, and
`mix-075`. No generation, scoring, training, network operation, commit, or
push was performed in this session. The historical 24-image diagnostic
benchmark contract remains untouched.

## Interpolation contract

- Endpoints remain pinned real checkpoints:
  - `parent-4000`:
    `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt`
    (step 4000, pinned SHA-256).
  - `finish-control-a4300`:
    `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt`
    (step 4300, pinned SHA-256).
- `mix-025`, `mix-050`, and `mix-075` evaluate respectively
  `(1-alpha) * parent-4000 + alpha * finish-control-a4300`, with alpha
  `0.25`, `0.50`, and `0.75`.
- The evaluator reads and blends only `state["model"]`, the checkpoint's
  serialized control/LoRA trainable model tensors. Optimizer, scheduler, RNG,
  counters, and other training metadata are never interpolated or emitted as a
  mixed checkpoint.
- Endpoint key sets, tensor-only values, floating dtypes, and shapes must
  match exactly. Blending is CPU FP32 and the result is cast to the parent
  tensor's evaluation dtype before strict model loading.
- Mix provenance records the exact candidate ID, alpha, formula, endpoint
  paths, embedded endpoint steps, and endpoint SHA-256 values in
  `final_val_provenance.json` and `checkpoint_preflight.json`. Preflight also
  validates those endpoint identities and trainable key/shape compatibility.
- Mix output images are named `<candidate>.png` (for example `mix-025.png`),
  rather than being presented as a real checkpoint step. Real-candidate image
  names and existing contracts remain `step_XXXXXX.png`.
- The locked evaluation settings remain 8 Turbo steps, CFG 0, mu 1.15, and
  control scale 1.0. No Turbo base/zero-adapter support was added.

## Files changed this session

- `scripts/final_val_turbo_benchmark.py`
- `tests/test_final_val_turbo_benchmark.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_final_val_turbo_benchmark -v
# 13 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/final_val_turbo_benchmark.py tests/test_final_val_turbo_benchmark.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/final_val_turbo_benchmark.py --help
git diff --check
```

Focused coverage proves FP32 blend/cast behavior, exact key/shape rejection,
endpoint provenance for `mix-025`, and model-only interpolation with
optimizer/scheduler state excluded. The CLI lists only the two real candidates
plus `mix-025`, `mix-050`, and `mix-075`.

## Exact GH200 commands

Run from the repository root on the writable GH200 host shell. Each candidate
uses its own new output root. Do not run `generate` unless the interpolation
evaluation is intentionally being executed.

```bash
uv run python scripts/final_val_turbo_benchmark.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-025
uv run python scripts/final_val_turbo_benchmark.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-025
uv run python scripts/final_val_turbo_benchmark.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-025 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/final_val_turbo_benchmark.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-025

uv run python scripts/final_val_turbo_benchmark.py preflight --candidate mix-050 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-050
uv run python scripts/final_val_turbo_benchmark.py generate --candidate mix-050 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-050
uv run python scripts/final_val_turbo_benchmark.py score --candidate mix-050 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-050 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/final_val_turbo_benchmark.py report --candidate mix-050 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-050

uv run python scripts/final_val_turbo_benchmark.py preflight --candidate mix-075 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-075
uv run python scripts/final_val_turbo_benchmark.py generate --candidate mix-075 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-075
uv run python scripts/final_val_turbo_benchmark.py score --candidate mix-075 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-075 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/final_val_turbo_benchmark.py report --candidate mix-075 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/mix-075
```

## Exact next task

On the writable GH200 host, run the three `preflight` commands and inspect the
new provenance files. If they validate the locked endpoint identities, a later
authorized evaluation session may run the staged generate/score/report commands
for the three mix candidates. Do not alter the frozen benchmark inputs or run
training.
