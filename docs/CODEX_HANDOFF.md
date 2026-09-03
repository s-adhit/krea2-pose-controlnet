# Project handoff

## Current objective

The isolated Krea-2 Turbo `mix-025` Pose-LoRA + one Style-LoRA composition matrix is implemented and has not been run on GH200. No training, network access, commit, push, canonical `inference.py` edit, training-code edit, or frozen final-val/prompt-injection/prompting-study artifact edit occurred.

## Locked composition contract

- Entry point: `scripts/style_lora_composition.py`; staged actions are `audit`, `preflight`, `generate`, `score`, `report`, and read-only `summary`.
- Candidate remains `mix-025`: FP32 `(0.75 * parent-4000) + (0.25 * finish-control-a4300)` over only `state['model']` trainable tensors, using the existing strict final-val interpolation implementation. Both pinned endpoint hashes are revalidated on every staged action.
- Runtime is Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, no resolution-dependent mu shift, native/aspect-preserving cached-latent geometry, control scale 1.0, and the same frozen final-val sampling seed per pose across every style. No source RGB is sampled, copied, or used as fallback.
- New immutable input: `docs/evaluation/style-lora-composition/style_lora_composition_v1.jsonl`, SHA-256 `cf3ac68a5500b5ab2938349b8eb74db1a6f711c9ee7f49c97e629beeccab52cb`. It contains exactly the requested four conditions and supportive frozen semantic prompts from the completed prompting guide. The adapters declare no trigger wording, so the prompt remains identical from pose-only through every style comparison.
- Initial variants are pose-only, darkbrush 1.0, rainywindow 1.0, retroanime 1.0, and realism 1.0. The runner excludes realism automatically if its strict audit fails; it never guesses a partial mapping.

## Adapter audit conclusions

- All four pinned local Style-LoRA files passed strict audit: 528 FP32 tensors, 264 exact A/B pairs, rank 32, and every target shape agrees with the shared Krea-2 Turbo MMDiT structure.
- `darkbrush`, `rainywindow`, and `retroanime` use `transformer.*`; the explicit mapping is `transformer_blocks -> blocks`, Krea `to_[q/k/v/gate]` and `to_out.0` -> `w[q/k/v]/gate/wo`, `ff -> mlp`, plus exact mappings for image input, final layer, time/text projections, and text fusion. Each has no safetensors metadata, so the runtime uses precisely `style_strength * (B @ A)` with no invented alpha/rank scaling.
- `realism` uses `base_model.model.*`. Removing only that prefix yields a deterministic one-to-one mapping to the same 264 runtime targets; all shapes validate. Its metadata declares `lora_rank=32` and `lora_alpha=32`, therefore its effective multiplier is 1.0 and delta is `style_strength * (32 / 32) * (B @ A)`. Realism is supported.
- Style tensors are separate from pose state and use temporary forward hooks. Hooks add low-rank outputs without changing model parameters, are scoped to one generation, and are always removed in `finally`; strength zero installs no hook. Style tensors are never merged, concatenated, or interpolated with pose checkpoint tensors.

## Files changed this session

- `pose_controlnet/style_lora.py` — strict file audit, namespace mapping, shape validation, adapter loader, reversible hooks.
- `scripts/style_lora_composition.py` — isolated four-pose staged matrix, provenance/audits/metadata, PCK+CLIP scoring, grids/contact sheet, style metrics, summary.
- `docs/evaluation/style-lora-composition/style_lora_composition_v1.jsonl` — new frozen experiment spec only.
- `tests/test_style_lora_composition.py` — frozen hashes, A/B/rank/shapes, target resolution, unsupported realism failure, scaling/no-leak, seed/control/prompt and metadata provenance.
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/style_lora_composition.py audit --output-root /tmp/style-lora-audit
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_style_lora_composition tests.test_chinese_prompt_smoke tests.test_prompting_guide_study tests.test_frozen_prompt_turbo tests.test_final_val_turbo_benchmark -v
# 38 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile pose_controlnet/style_lora.py scripts/style_lora_composition.py tests/test_style_lora_composition.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/style_lora_composition.py --help
git diff --check
```

## Exact GH200 commands

Output root (new and isolated): `/lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0`

```bash
uv run python scripts/style_lora_composition.py audit --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py preflight --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py generate --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py score --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/style_lora_composition.py report --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
```

Exact compact summary command:

```bash
uv run python scripts/style_lora_composition.py summary --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
```

## Exact next task

Run and inspect the 4-pose Style-LoRA matrix. Do not start a strength sweep or combine two Style-LoRAs yet.
