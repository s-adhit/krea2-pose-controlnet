# Phase 1 handoff

## Current bounded objective and status

The timestep-exposure continuation completed through step 1800. Its Turbo
evaluation is implemented but has **not** been run: no Turbo preflight,
generation, scoring, training, resume, checkpoint mutation, HF upload,
commit, or push occurred in this session.

The new read-only staged entrypoint is:

```bash
scripts/turbo_timestep_benchmark.py
```

It is isolated from the original and LR-only Turbo trees and exposes only
`preflight`, `generate`, `score`, and `report`.

## Immutable Turbo evaluation contract

- Base: Krea-2 Turbo.
- Inference remains exactly 8 steps, CFG `0.0`, `mu=1.15`, with
  `mu_resolution_dependent=false` and the existing official schedule.
- Diagnostic samples remain the exact existing 24 stems, prompts, controls,
  bucket/crop geometry, sample identities, and seeds (fixed seed `420200`).
- PCK remains `score_authoritative_pck` with confidence threshold `0.5`; CLIP
  remains the shared `_clip_score` implementation.
- The evaluation entrypoint cannot construct an optimizer, backward pass, or
  training operation.

## Exact checkpoint identities

HF repo:

```text
adhit-420/Krea-2-PoseControl-LoRA-checkpoints
```

Only this namespace is accepted:

```text
pose-learning-1500-timestep-lowmid20-to1800/full/
```

Only these exact checkpoints are accepted:

```text
step_001600.pt
step_001700.pt
step_001800.pt
```

Steps 1600 and 1700 use `validated_hf_checkpoint_for_step` only. Local step
1800 uses `validated_local_checkpoint_for_hf_step`, which requires its exact
HF `.complete.json` marker, SHA-256, complete training-state
deserialization/schema, and matching embedded `global_step`. Neither path has
a nearest/latest/timed-mirror/original-branch/LR-only fallback. The run root
is fixed to:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800
```

The local step-1800 checkpoint is at:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800/step_001800.pt
```

Steps 1600 and 1700 were pruned locally after their successful mirrors;
preflight recovers only their exact completion-marked HF paths.

## User-run commands

Run from the GH200 host only when evaluation is authorized:

```bash
cd /home/ubuntu/Krea-2-Pose-ControlNet
export UV_CACHE_DIR=/tmp/krea_uv_cache
uv run python scripts/turbo_timestep_benchmark.py preflight
uv run python scripts/turbo_timestep_benchmark.py generate
uv run python scripts/turbo_timestep_benchmark.py score
uv run python scripts/turbo_timestep_benchmark.py report
```

Evaluation output is fixed to:

```text
/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-timestep-lowmid20
```

It rejects both:

```text
/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0
/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-lr5e5
```

Expected outputs are `turbo_spec.json`, `checkpoint_preflight.json`, per-stem
`fixed_pose/<stem>/control.png`, `metadata.json`, and step-1600/1700/1800
PNGs, `generation_results.json`, `pck_clip_results.json`,
`turbo_timestep_checkpoint_selection_grid.png`,
`turbo_timestep_full_contact_sheet.png`, and `evaluation_summary.json`.
The report reads existing machine-readable original step-900 @ `1e-4` and
LR-only step-1500 @ `5e-5` results, then reports those plus timestep 1600,
1700, and 1800; it does not recompute either baseline.

## Files changed and validation

- `pose_controlnet/turbo_evaluation.py`: strict timestep namespace/step/output
  guards and shared diagnostic-contract assertion.
- `scripts/turbo_timestep_benchmark.py`: staged read-only Turbo evaluation.
- `tests/test_turbo_timestep_evaluation.py`: focused acceptance/rejection,
  immutable Turbo/PCK/CLIP/input contract, and no-training coverage.
- `docs/CODEX_HANDOFF.md`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile
pose_controlnet/turbo_evaluation.py scripts/turbo_timestep_benchmark.py
tests/test_turbo_timestep_evaluation.py`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest
tests.test_train_mechanics tests.test_turbo_evaluation
tests.test_turbo_lr5e5_evaluation tests.test_turbo_timestep_evaluation` — 69 tests.

No generation, score, preflight, report, training, or checkpoint operation was
executed by Codex. Before ending this session: run `git diff --check` and
`git status --short`.
