# Project handoff

## Current objective

The isolated English-versus-Chinese prompt smoke test is implemented but has
not been run on GH200. No training, network access, commit, or push occurred.
It is separate from all 64-row prompting-study artifacts and aggregates.

## Frozen language-smoke contract

- Entry point: `scripts/chinese_prompt_smoke.py` with staged `preflight`,
  `generate`, `score`, `report`, and read-only `summary` actions.
- Frozen input: `docs/evaluation/prompting-guide/chinese_prompt_smoke.jsonl`.
  Byte SHA-256: `c782d6fecff1bc6393f9175a52cb9b66f11185dcf0a3a3c8cccf1ab3a095769e`.
- It accepts exactly two ordered UTF-8 JSONL rows, with exact languages `en`
  then `zh`, both for `sculpture_humanart_14000000003803`. Exact prompt text
  is recorded in provenance, generation metadata, scores, and compact output.
- Candidate is hard-locked to `mix-025`; the authoritative final-val sampling
  seed is resolved once and must be identical for both rows. The authoritative
  pose control is resolved through the final-val DatasetIndex and SHA-256
  pinned. Native/aspect-preserving cached-latent geometry and Turbo settings
  remain locked: 8 steps, CFG 0, mu 1.15, control scale 1.0.
- Generation fails closed on input-hash/schema/language/stem/provenance/seed/
  control-hash/geometry/Turbo drift or incomplete artifacts. It generates
  exactly two images. No source RGB is read, copied, or used as fallback.
- Scoring uses existing unchanged authoritative PCK and CLIP behavior, once
  per language. Reporting writes `metrics_by_language.json`, a compact summary,
  and `english_vs_chinese_comparison.png` with `pose control | English |
  Chinese` columns.

## Files changed this session

- `scripts/chinese_prompt_smoke.py` (new isolated staged evaluation tool)
- `tests/test_chinese_prompt_smoke.py` (new focused fail-closed tests)
- `docs/CODEX_HANDOFF.md`

The supplied frozen language-smoke JSONL is untracked and was not modified.
The frozen 64-row prompting study, its SHA/results, prompt-injection benchmark,
final-val benchmark/spec/sidecar, canonical `inference.py`, and training code
are untouched.

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_chinese_prompt_smoke tests.test_prompting_guide_study tests.test_frozen_prompt_turbo tests.test_final_val_turbo_benchmark -v
# 33 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/chinese_prompt_smoke.py tests/test_chinese_prompt_smoke.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/chinese_prompt_smoke.py --help
git diff --check
```

Focused coverage pins the frozen UTF-8 SHA and exact rows; rejects hash drift,
wrong languages/stem, and incomplete outputs; requires ordered language score
records; and proves both metadata records share seed/control/candidate/native
geometry/locked Turbo provenance. Existing prompting-guide, prompt-injection,
and final-val benchmark tests remain green.

## Exact GH200 commands

Run from the repository root in the GH200 host shell. Use this new root only;
do not write into frozen final-val, prompt-injection, or 64-row study roots.

```bash
uv run python scripts/chinese_prompt_smoke.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/chinese-smoke-mix-025
uv run python scripts/chinese_prompt_smoke.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/chinese-smoke-mix-025
uv run python scripts/chinese_prompt_smoke.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/chinese-smoke-mix-025 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/chinese_prompt_smoke.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/chinese-smoke-mix-025
```

Exact compact summary command:

```bash
uv run python scripts/chinese_prompt_smoke.py summary --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/chinese-smoke-mix-025
```

## Exact next task

Inspect EN-vs-ZH result, then write root-level `prompting.md`.
