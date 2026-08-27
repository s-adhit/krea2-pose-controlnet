# Phase 1 handoff

## Current bounded objective and status

Evaluation support is complete for the finished LR=5e-5 continuation branch.
No training, training resume, checkpoint creation/change, HF upload, W&B run,
commit, push, GH200 generation, detector scoring, or CLIP scoring was started
in this session.

## Completed LR continuation branch

- Run name and sole allowed HF namespace:
  `pose-learning-900-lr5e5-to1500/full/` in
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`.
- Exact evaluated steps are only `1000, 1100, 1200, 1300, 1400, 1500`.
- The evaluator rejects any different step list, local checkpoint root, or HF
  repository. It resolves every checkpoint only through
  `validated_hf_checkpoint_for_step` for
  `pose-learning-900-lr5e5-to1500/full/step_XXXXXX.pt` and its matching
  `.complete.json` marker. That existing validator preserves exact-marker,
  SHA-256, full deserialization/schema, and embedded `global_step` validation.
  No original `pose-learning-1500`, timed, nearest, latest, or other namespace
  fallback is permitted.
- Exact local checkpoint root:
  `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500`.
- Exact isolated evaluation root:
  `/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-lr5e5`.
  The evaluator refuses both the canonical RAW evaluation path and the original
  Turbo path `/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0`.

## Evaluation contract

- Entry point: `scripts/turbo_lr5e5_benchmark.py` with `preflight`,
  `generate`, `score`, and `report` subcommands only. It does not construct an
  optimizer or call backward/training APIs.
- Turbo sampling is unchanged: Krea-2 Turbo, 8 steps, CFG 0.0, `mu=1.15`, no
  resolution-dependent shift, exact existing official schedule, unchanged
  sampler/control loading/VAE/decode.
- It derives the diagnostic spec with the existing Turbo implementation and
  verifies all 24 stems, cached input identities, and per-stem seeds against
  the original Turbo `turbo_spec.json` before any output write.
- PCK uses the unchanged `score_authoritative_pck` implementation; CLIP uses
  the exact existing `scripts.turbo_benchmark._clip_score` function. This
  retains 21 authoritative Human-Art/COCO samples, 3 unavailable Danbooru
  exclusions, renderer-qualified COCO-17 PCK, confidence 0.5, deterministic
  Hungarian association, reference bbox-diagonal normalization, and unmatched
  reference-person failures.
- `report` reads but does not recompute or overwrite original step-900
  results/images. It compares original `900 @ 1e-4` with new `1000..1500 @
  5e-5`, reports pooled/single/multi/COCO/Human-Art PCK at all three
  thresholds plus coverage/person counts/CLIP, and emits compact and full
  qualitative contact sheets.

## Files changed this session

- `pose_controlnet/turbo_evaluation.py`
- `scripts/turbo_lr5e5_benchmark.py`
- `tests/test_turbo_lr5e5_evaluation.py`
- `docs/CODEX_HANDOFF.md`

## Verified checks

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile pose_controlnet/turbo_evaluation.py scripts/turbo_lr5e5_benchmark.py tests/test_turbo_evaluation.py tests/test_turbo_lr5e5_evaluation.py`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_turbo_evaluation tests.test_turbo_lr5e5_evaluation` — 19 tests.
- PASS: `git diff --check`.
- No live GH200 preflight/generation/scoring/report command was run.

## Exact future GH200 commands

Run sequentially from the GH200 host only after reviewing each prior output:

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
cd /home/ubuntu/Krea-2-Pose-ControlNet
uv run python scripts/turbo_lr5e5_benchmark.py preflight
uv run python scripts/turbo_lr5e5_benchmark.py generate
uv run python scripts/turbo_lr5e5_benchmark.py score
uv run python scripts/turbo_lr5e5_benchmark.py report
```

Expected branch-only outputs below
`/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-lr5e5`:
`turbo_spec.json`, `checkpoint_preflight.json`, `generation_results.json`,
`pck_clip_results.json`, `evaluation_summary.json`, per-stem `fixed_pose`
controls/images/metadata, `turbo_lr5e5_checkpoint_selection_grid.png`, and
`turbo_lr5e5_full_contact_sheet.png`.

## Next action

Stop here. The bounded implementation task is complete. If live evaluation is
authorized later, begin with the preflight command only; do not train or
resume training.
