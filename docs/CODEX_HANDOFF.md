# Project handoff

## Current objective

Run and inspect the trigger-corrected, isolated Krea-2 Turbo `mix-025` Pose-LoRA + one Style-LoRA matrix. Do not train, access the network from Codex, commit, push, alter canonical `inference.py`, alter training code, alter v1, or start a Style-LoRA strength sweep.

## Composition contract

- Candidate: `mix-025`, FP32 `(0.75 * parent-4000) + (0.25 * finish-control-a4300)` over only `state['model']` trainable tensors. Both pinned endpoint hashes are revalidated for every stage.
- Runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, no resolution-dependent mu shift, native/aspect-preserving cached-latent geometry, control scale 1.0, and frozen final-val sampling seeds. Source RGB is neither sampled nor used as fallback.
- Matrix: four poses (`simple_single`, `dynamic_airborne`, `inversion`, `multi_person`) × pose-only, darkbrush@1.0, rainywindow@1.0, retroanime@1.0, realism@1.0. PCK and CLIP retain the existing metric semantics; the CLIP text is the exact prompt used to generate each matrix cell.
- Style tensors remain separate from Pose-LoRA state and are applied only in temporary scoped hooks. All four pinned adapters must pass the strict 528-FP32-tensor / 264-pair / rank-32 mapping audit.

## Historical v1 and trigger-correct v2

- Preserve `docs/evaluation/style-lora-composition/style_lora_composition_v1.jsonl` unchanged: SHA-256 `cf3ac68a5500b5ab2938349b8eb74db1a6f711c9ee7f49c97e629beeccab52cb`.
- Preserve the completed `mix-025-strength-1.0` results root/artifacts unchanged. It is historical **no-trigger composition sanity evidence only**, not a valid style-fidelity experiment: its frozen spec supplied no required official Style-LoRA trigger wording.
- New immutable v2 spec: `docs/evaluation/style-lora-composition/style_lora_composition_v2_triggers.jsonl`, SHA-256 `89916989cdf8bc083cf868793647cf7239313ed7e8cc4badabb44eb40a13736d`.
- v2 prompt construction is frozen per cell in immutable provenance as separate `semantic_base_prompt`, `trigger_phrase`, and `effective_prompt` fields. It uses the exact prior semantic base prompts, unmodified except for the exact required suffix where applicable:
  - pose-only: no trigger; effective prompt is the semantic base prompt.
  - realism: no trigger; effective prompt is the semantic base prompt.
  - darkbrush: `, monochrome ink wash style`.
  - rainywindow: `, rainy window style`.
  - retroanime: `, Purple retro anime style`.
- `scripts/style_lora_composition.py` selects the new spec only with `--experiment v2-triggers`; the default explicit legacy mode remains `v1`, retaining its old no-trigger metadata/provenance interpretation. Named specs and their SHA checks fail closed. A v1 and v2 contract have distinct kind/spec/prompt provenance, so attempting to use one output root for both fails closed.

## Files changed this session

- `docs/evaluation/style-lora-composition/style_lora_composition_v2_triggers.jsonl`
- `scripts/style_lora_composition.py`
- `tests/test_style_lora_composition.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/style_lora_composition.py tests/test_style_lora_composition.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_style_lora_composition -v
# 8 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/style_lora_composition.py --help
```

The composition tests cover frozen v1/v2 SHA validation; exact official trigger phrases; no trigger for pose-only and realism; unchanged semantic prompts across all variants; deterministic effective prompts; Style-LoRA audit and scoped-hook safety; staged immutable provenance; and v1/v2 output-identity collision failure.

## Exact GH200 commands

Use the new isolated root only:

```bash
uv run python scripts/style_lora_composition.py audit --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers
uv run python scripts/style_lora_composition.py preflight --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers
uv run python scripts/style_lora_composition.py generate --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers
uv run python scripts/style_lora_composition.py score --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/style_lora_composition.py report --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers
uv run python scripts/style_lora_composition.py summary --experiment v2-triggers --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0-triggers
```

The report produces four per-pose grids, an aggregate contact sheet, PCK/CLIP outputs grouped by style, immutable provenance, and a compact summary. Never target the historical `mix-025-strength-1.0` root.

## Next action

Run and inspect the trigger-corrected 4-pose matrix above. Do not begin a style-strength sweep.
