# Project handoff

## Current objective

The v1 release decision is frozen as documentation/provenance only. The exact
next task is **canonical `inference.py` integration against this frozen release
contract**. Do not alter the decision artifact without deliberately updating
its pinned test hash.

## Frozen release decision

- Machine-readable contract:
  `docs/evaluation/release/final_release_v1.json`
- Contract SHA-256:
  `9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`
- Human summary: `docs/evaluation/release/FINAL_RELEASE_DECISION.md`
- Integrity test: `tests/test_final_release_decision.py`
- Candidate: `mix-025`, float32 interpolation of `state['model']` trainable
  control/LoRA tensors only, alpha `0.25`:
  `(1 - alpha) * parent-4000 + alpha * finish-control-a4300`.
- Parent SHA-256:
  `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd`.
- A4300 SHA-256:
  `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff`.
- Canonical runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, no
  resolution-dependent `mu`; native aspect-preserving cached latent bucket;
  control scale `1.0`. Dynamic-768 remains optional; `1.25-1.50` is optional
  stronger control.
- Canonical Style-LoRA scope: one adapter at a time, separate from Pose-LoRA,
  no permanent merge, reversible runtime application, configurable strength;
  no multi-Style-LoRA composition. Defaults: darkbrush `0.75`, rainywindow
  `0.50`, retroanime `0.50`, realism `0.25`.

## Evidence and known limitations

- The release contract pins all referenced local frozen specs/results and their
  available SHA-256 values, including final-val, Turbo baseline, native vs
  dynamic-768, control scale, hard-pose, hands, and Style-LoRA sweep evidence.
- Mix-025 wins the frozen final-val PCK comparison; native geometry is the
  default; control scale remains `1.0` despite aggregate best PCK at `1.5`
  because control behavior is non-monotonic by pose class.
- Overlapping multi-person interactions remain a limitation. No systematic
  Pose-Control-specific hand regression was observed. Style strength is
  non-monotonic.

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile tests/test_final_release_decision.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_final_release_decision -v
# 6 tests passed
git diff --check
sha256sum docs/evaluation/release/final_release_v1.json
# 9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b
```

No network access, generation, training, inference.py/training-code changes,
commit, or push occurred. No project master TODO/Section M file exists in the
tracked repository, so none was changed.

## Files changed this session

- `docs/evaluation/release/final_release_v1.json`
- `docs/evaluation/release/FINAL_RELEASE_DECISION.md`
- `tests/test_final_release_decision.py`
- `docs/CODEX_HANDOFF.md`

## Still incomplete

- Canonical `inference.py` integration.
- README finalization.
- Final hero/showcase.
- Exact report metadata/formulas.
- Attribution tasks.
