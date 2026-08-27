# Phase 1 handoff

## Current status and key result

The completed timestep-exposure continuation was negative under the immutable
Turbo contract. The best balanced result remains the LR-only branch at step
1500, not timestep steps 1600/1700/1800. At step 1500: CLIP
`0.33684297981215466`, detection coverage `0.9230769230769231`, joint coverage
`0.9271844660194175`, PCK@0.05/0.10/0.20 `0.05461165048543689` /
`0.1808252427184466` / `0.41262135922330095`, and four unmatched reference
people. The 1600–1800 branch reduced CLIP, detection, joint coverage, and
PCK@0.20 (to `0.3993`, `0.3993`, and `0.3786` respectively), so no later
timestep checkpoint should be used as the current diagnostic source.

The current bounded milestone is implemented but not run: two read-only
diagnostics for only this LR-only step-1500 source. No training, resume,
checkpoint mutation, HF upload, commit, push, generation, score, preflight, or
projection audit was executed in this session.

## Immutable source and Turbo contract

Source local checkpoint:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt
```

It is accepted only after validating its local file against the exact HF
completion marker/SHA-256/schema/embedded step in:

```text
adhit-420/Krea-2-PoseControl-LoRA-checkpoints
pose-learning-900-lr5e5-to1500/full/
```

There is no nearest/latest, remote-payload, timed-mirror, original-branch, or
timestep-branch fallback. Turbo remains Krea-2 Turbo, eight steps, CFG `0.0`,
`mu=1.15`, `mu_resolution_dependent=false`, and the existing official schedule.
All 24 existing diagnostic stems, prompts, controls, per-stem seeds, buckets,
paired geometry, VAE/decode, authoritative PCK (`confidence_threshold=.5`), and
shared CLIP implementation are retained.

## Newly implemented diagnostics

`scripts/turbo_control_scale_sweep.py` exposes only `preflight`, `generate`,
`score`, and `report`. Its sole inference change is optional `control_scale` in
`sample_turbo_pose_image`: it multiplies only the clean control latent
immediately before the usual control patchification/concat path. The fixed
scales are `0.75, 1.00, 1.25, 1.50, 2.00`. The default `1.0` returns the
original control tensor itself, preserving established baseline semantics. The
output root is isolated at:

```text
/lambda/nfs/adhit/krea2-pose/evaluation/turbo-control-scale-step1500
```

It rejects the original, LR-only, and timestep Turbo trees. Expected files are
`turbo_spec.json`, `checkpoint_preflight.json`, per-stem controls,
scale-specific metadata and PNGs, `generation_results.json`,
`pck_clip_results.json`, `turbo_control_scale_selection_grid.png`,
`turbo_control_scale_full_contact_sheet.png`, and `evaluation_summary.json`.

`scripts/audit_control_projection.py` exposes `preflight` and `audit`. It uses
the exact Raw model recorded in the step-1500 training state and directly
invokes its learned `model.first` (`ControlInputLayer`) using cached real
diagnostic latents across their actual buckets. Executable ordering is verified
as `[noisy_image_patch_tokens, clean_control_patch_tokens]`; BF16 conversion and
patchification occur first, while positional encoding and normalization occur
after this projection. At deterministic noise and timesteps `0.1, 0.3, 0.5,
0.7, 0.9`, it reports image/control RMS and L2 plus actual image-only
`[image, 0]`, control-only `[0, control]`, and both `[image, control]`
projection output RMS/L2. The aggregate ratio is
`control_only_projection_output_rms / image_only_projection_output_rms`.
Output is:

```text
/lambda/nfs/adhit/krea2-pose/evaluation/control-projection-step1500/control_projection_audit.json
```

## Exact user-run commands

Run on the GH200 host only when authorized:

```bash
cd /home/ubuntu/Krea-2-Pose-ControlNet
export UV_CACHE_DIR=/tmp/krea_uv_cache
uv run python scripts/turbo_control_scale_sweep.py preflight
uv run python scripts/turbo_control_scale_sweep.py generate
uv run python scripts/turbo_control_scale_sweep.py score
uv run python scripts/turbo_control_scale_sweep.py report
uv run python scripts/audit_control_projection.py audit
```

The projection `audit` command writes the stated JSON. Its optional
`preflight` verifies source identity, Raw provenance, multiple actual bucket
shapes, and the fixed timestep grid without loading the large model.

## Files and validation in this session

- `pose_controlnet/turbo_evaluation.py`: optional identity-default control
  scale, fixed scale/output helpers, exact local step-1500 validator.
- `scripts/turbo_control_scale_sweep.py`: staged read-only Turbo sweep.
- `scripts/audit_control_projection.py`: staged read-only actual-projection
  magnitude audit.
- `tests/test_control_diagnostics.py`: control-only scale, baseline identity,
  fixed contract/source, feature ordering, real projection, deterministic grid,
  and no-training coverage.
- `docs/CODEX_HANDOFF.md`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile
pose_controlnet/turbo_evaluation.py scripts/turbo_control_scale_sweep.py
scripts/audit_control_projection.py tests/test_control_diagnostics.py`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest
tests.test_turbo_evaluation tests.test_turbo_lr5e5_evaluation
tests.test_control_diagnostics` — 29 tests.

Before handing off, run `git diff --check` and `git status --short`. Do not
run either GH200 diagnostic or production training without explicit approval.
