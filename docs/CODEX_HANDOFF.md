# Project handoff

## Current objective

Frozen prompt-injection evaluation and same-pose hero generation support are
implemented as a separate opt-in evaluator. No generation, scoring, training,
network operation, commit, or push was performed in this session.

## New frozen contracts

- Entry point: `scripts/frozen_prompt_turbo.py`.
- Modes are separate from the source-caption final-val evaluator:
  - `prompt-injection {preflight,generate,score,report}`;
  - `hero {preflight,generate,report}`. Hero rejects `score` because it is
    generation-only.
- Candidate defaults to `mix-025`, while allowlisting and reusing the existing
  `parent-4000`, `finish-control-a4300`, `mix-025`, `mix-050`, and `mix-075`
  candidate/interpolation implementation from `final_val_turbo_benchmark.py`.
  Interpolation provenance remains endpoint paths, embedded steps, pinned
  SHA-256 values, alpha, and the model-only FP32 blend formula.
- Prompt injection verifies byte SHA-256
  `a7c6f3aa8aa1e18bc0767b9ad940b1c0d33fbabdfcdf568cffabb883b605bdf3`,
  exact five-field schema, exactly 48 unique stems, and exact frozen final-val
  stem order before any output is touched. It validates the original cached
  final-val identity, then replaces only runtime Qwen text conditioning with
  the frozen injected prompt. The original source-caption benchmark is not
  modified or used as a sampling fallback.
- Prompt injection uses the frozen final-val controls, frozen per-stem sampling
  seeds, native/aspect-preserving cached-latent bucket geometry, and locked
  Turbo 8 steps / CFG 0 / mu 1.15 / control scale 1.0. PCK uses the canonical
  v3 sidecar; CLIP receives the injected prompt text. Provenance records the
  exact prompt mapping, prompt/control SHA values, final spec SHA, seeds,
  geometry, candidate, and sampler settings. Incomplete/corrupt/mismatched artifact sets fail
  closed. Reports contain only pose control + candidate output contact sheets.
- Hero verifies byte SHA-256
  `1b28d8b9cc8754327727a317de03543aa71876ba0f878acd0ad8dc45897e9345`,
  exact three-field schema, six unique hero IDs, and the sole canonical stem
  `real_human_humanart_15000000000930`. Every interpretation starts from that
  same frozen latent/control geometry; per-prompt sampling seeds are derived
  deterministically as `SHA256('420600:<hero_id>:sampling')[:8] mod (2^63-1)`.
  Metadata and the hero summary retain prompt, seed, candidate/interpolation,
  sampler settings, prompt/control hashes, and native geometry. The hero contact sheet
  is pose control plus six generated outputs only; source RGB fallback is
  explicitly prohibited.

## Files changed this session

- `scripts/frozen_prompt_turbo.py`
- `tests/test_frozen_prompt_turbo.py`
- `docs/CODEX_HANDOFF.md`

Frozen final-val benchmark/spec/sidecar and all prior final-val results were
not modified.

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_frozen_prompt_turbo tests.test_final_val_turbo_benchmark -v
# 19 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/frozen_prompt_turbo.py tests/test_frozen_prompt_turbo.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/frozen_prompt_turbo.py --help
git diff --check
```

Focused coverage verifies pinned prompt loading, hash drift rejection,
duplicate/wrong-order rejection, canonical six-prompt hero binding,
deterministic unique hero seeds, and fail-closed incomplete generation status.
The existing final-val suite remains green.

## Exact GH200 commands

Run from the repository root in the writable GH200 host shell. Use new output
roots; do not place these artifacts in source-caption final-val directories.

```bash
uv run python scripts/frozen_prompt_turbo.py prompt-injection preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompt-injection/mix-025
uv run python scripts/frozen_prompt_turbo.py prompt-injection generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompt-injection/mix-025
uv run python scripts/frozen_prompt_turbo.py prompt-injection score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompt-injection/mix-025 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/frozen_prompt_turbo.py prompt-injection report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompt-injection/mix-025

uv run python scripts/frozen_prompt_turbo.py hero preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hero-same-pose/mix-025
uv run python scripts/frozen_prompt_turbo.py hero generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hero-same-pose/mix-025
uv run python scripts/frozen_prompt_turbo.py hero report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hero-same-pose/mix-025
```

## Exact next task

Style-LoRA composition support.
