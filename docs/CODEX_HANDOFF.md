# Project handoff

## Current objective

Repair the isolated Krea-2 Turbo `mix-025` Pose-LoRA + one Style-LoRA composition runner’s staged provenance lifecycle. No training, network access, commit, push, canonical `inference.py` edit, training-code edit, or frozen benchmark-artifact edit occurred.

## Locked composition contract

- Entry point: `scripts/style_lora_composition.py`; staged actions are `audit`, `preflight`, `generate`, `score`, `report`, and read-only `summary`.
- Candidate remains `mix-025`: FP32 `(0.75 * parent-4000) + (0.25 * finish-control-a4300)` over only `state['model']` trainable tensors. Both pinned endpoint hashes are revalidated on every staged action.
- Runtime is Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, no resolution-dependent mu shift, native/aspect-preserving cached-latent geometry, control scale 1.0, and the frozen final-val sampling seed per pose across styles. No source RGB is sampled, copied, or used as fallback.
- Frozen input: `docs/evaluation/style-lora-composition/style_lora_composition_v1.jsonl`, SHA-256 `cf3ac68a5500b5ab2938349b8eb74db1a6f711c9ee7f49c97e629beeccab52cb`; exactly four pose conditions and supportive frozen semantic prompts. Initial variants are pose-only, darkbrush, rainywindow, retroanime, and realism at strength 1.0.
- All four pinned Style-LoRA files previously passed strict audit: 528 FP32 tensors, 264 exact A/B pairs, rank 32, complete runtime target mapping. Style tensors remain separate from pose state and use temporary scoped hooks only.

## Provenance fix

Root cause of the GH200 `preflight -> generate` failure: `StyleLoRAAudit.json()` contains `errors` as a Python tuple. The old runner wrote the broad live audit dict to JSON (where the tuple becomes a list), then compared the later deserialized JSON object to a freshly reconstructed live dict. This produced a false immutable-provenance mismatch even when every real experiment input was unchanged.

`style_lora_provenance.json` now contains one JSON-canonical immutable experiment payload, built and validated by every stage, including audit. It includes frozen-spec SHA, exact conditions/prompts, candidate interpolation identity and endpoint hashes, control hashes, seeds, native buckets/geometry, locked Turbo settings, style strength, and Style-LoRA hash/namespace/rank/mapping/scaling contract. Host- or stage-local paths and stage outputs are excluded.

Stage artifacts now store that payload under `immutable_provenance`; mutable fields such as action completion, artifact lists/paths, audit detail paths, training metadata, generated artifact lists, reports, and summary state are separate. Score additionally records immutable scoring provenance: canonical experiment-payload hash, reference-sidecar SHA, CLIP model id, and threshold. Existing provenance conflicts remain fail-closed.

## Files changed this session

- `scripts/style_lora_composition.py` — canonical immutable experiment and scoring provenance, strict validation shared by all stages, and mutable stage-payload separation.
- `tests/test_style_lora_composition.py` — staged lifecycle and immutable-drift regression coverage.
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/style_lora_composition.py tests/test_style_lora_composition.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_style_lora_composition -v
# 6 tests passed, including audit -> preflight -> generate -> score -> report -> summary provenance lifecycle,
# mutable state separation, and fail-closed frozen-spec/style-strength/Style-LoRA-hash/candidate drift.
```

## GH200 cleanup and deterministic rerun

The previous root has only stale buggy provenance and no completed generations. After deploying this code, remove only that isolated root:

```bash
rm -rf -- /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
```

Then rerun in this exact order:

```bash
uv run python scripts/style_lora_composition.py audit --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py preflight --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py generate --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py score --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/style_lora_composition.py report --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
uv run python scripts/style_lora_composition.py summary --candidate mix-025 --style-strength 1.0 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/style-lora-composition/mix-025-strength-1.0
```

## Next action

Run the isolated 4-pose matrix from the cleanup/rerun sequence above. Do not begin a strength sweep, combine two Style-LoRAs, or train.
