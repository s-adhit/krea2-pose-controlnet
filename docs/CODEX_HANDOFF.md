# Project handoff

## Current objective

Canonical public inference is implemented against frozen v1. The next action
is one manual GH200 smoke generation only; do not train, make a hero/showcase,
or change the frozen release artifact in that smoke task.

## Frozen release contract

- Contract: `docs/evaluation/release/final_release_v1.json`
- SHA-256: `9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`
- Candidate: `mix-025`, FP32 blend of trainable `state['model']` tensors only:
  `(1 - 0.25) * parent-4000 + 0.25 * finish-control-a4300`.
- Pinned endpoints: parent step 4000
  `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd`;
  finish-control step 4300
  `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff`.
- Runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0.
  Native/aspect-preserving is default; dynamic-768 is explicit opt-in.

## Implemented canonical inference behavior

- `inference.py` defaults to `mix-025`; it verifies the frozen JSON hash,
  canonical endpoint paths/hashes, embedded steps, compatible Raw provenance,
  matching trainable tensor keys/shapes, and FP32 interpolation before loading
  the Turbo model. `--parent-ckpt` and `--finish-ckpt` must be supplied as a
  pair and still must match the frozen endpoint hashes.
- Historical direct checkpoint use remains available through
  `--pose-lora-ckpt PATH` (and cannot be combined with canonical endpoints).
- Native input geometry retains the supplied pose canvas exactly and rejects
  non-16-aligned dimensions; it never silently selects dynamic-768. Use
  `--dynamic-768-bucket` for the alternate policy or `--width W --height H`
  for an explicit output canvas. Every sidecar records the mode and exact
  output bucket/dimensions.
- Style scope is zero or one adapter: `--style-name` selects only frozen-known
  defaults, `--style-lora PATH` requires a known name for a custom path, and
  `--style-strength` is non-negative/configurable. Audits fail closed;
  strength zero does not load an adapter or install hooks. No trigger phrase
  is injected; sidecars explicitly record the unchanged effective prompt.
- Sidecars include release ID/path/hash, candidate/interpolation/endpoints,
  pose hash, Turbo settings, control scale, geometry, and Style-LoRA data.

## CLI

```bash
python inference.py --turbo-ckpt PATH --prompt TEXT --pose-image POSE.png --output OUT.png
```

Optional release controls: `--candidate mix-025`, `--parent-ckpt PATH`,
`--finish-ckpt PATH`, `--control-scale FLOAT`, `--dynamic-768-bucket`,
`--width W --height H`, `--style-name {darkbrush,rainywindow,retroanime,realism}`,
`--style-lora PATH`, and `--style-strength FLOAT`.

## Files changed this session

- `inference.py`
- `pose_controlnet/trainable_interpolation.py`
- `scripts/final_val_turbo_benchmark.py` (now reuses the shared interpolation)
- `tests/test_inference.py`
- `README.md` (inference section only)
- `docs/CODEX_HANDOFF.md`

The pre-existing untracked frozen release files and release-decision test were
preserved; `final_release_v1.json` was not edited.

## Verification

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile inference.py pose_controlnet/trainable_interpolation.py scripts/final_val_turbo_benchmark.py tests/test_inference.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_inference tests.test_final_release_decision tests.test_style_lora_composition tests.test_final_val_turbo_benchmark -v
# 43 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python inference.py --help
sha256sum docs/evaluation/release/final_release_v1.json
# 9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b
```

No network access, training, full image generation, commit, or push occurred.

## Exact GH200 smoke command

Run manually from the GH200 host shell (this is a single generation, using the
known 300x387 repository control with the explicit dynamic-768 mode):

```bash
cd /home/ubuntu/krea2-pose-controlnet && PYTHONPATH=. UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python inference.py --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors --prompt "fantasy mage, ornate robes, cinematic lighting" --pose-image docs/assets/showcase/final/fantasy-mage/condition.png --output /tmp/krea2-pose-v1-smoke.png --seed 42 --dynamic-768-bucket
```

If that smoke passes, the exact next action is final hero/showcase generation.
