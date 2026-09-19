# Project handoff

## Current objective

Materialize the frozen final `krea2-pose-control-lora-v1` candidate `mix-025`
as a compact standalone public inference artifact. The reproducible path is
implemented and fully verified, but this Codex sandbox mounts
`/lambda/nfs/adhit/krea2-pose` read-only, so it could not publish to the
requested NFS release directory. No endpoint checkpoint, frozen contract,
benchmark artifact, HF repository, commit, or push was modified.

## Frozen input verification

PASS before materialization:

- `docs/evaluation/release/final_release_v1.json`:
  `9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`
- parent-4000: `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd`
- finish-control-a4300: `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff`

Both full source states have 450 float32 trainable control/LoRA tensors,
215,488,512 parameters, and identical recorded Krea-2 Raw provenance.

## Release artifact

- Intended public path (blocked only by this sandbox mount):
  `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors`
- Verified staged artifact:
  `/tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors`
- Adjacent provenance JSON:
  `/tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors.provenance.json`
- SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`
- Format: `safetensors`; direct model tensors only (no optimizer, scheduler,
  RNG, counters, or unrelated full-training state).
- Tensor count: 450; parameter count: 215,488,512; alpha: 0.25.

`scripts/materialize_final_release.py` validates all three frozen hashes
before deserializing endpoints, requires exact model keys/shapes/floating
tensors, computes `(0.75 * parent + 0.25 * A4300)` in float32, writes an
atomic no-overwrite artifact, writes provenance, reloads it, and compares every
saved tensor exactly against the FP32 interpolation. Header canonicalization
makes the safetensors byte deterministic: two independent artifacts were
byte-identical and had the SHA above.

`inference.py --release-artifact <artifact>` now loads this public compact
format through the normal control/LoRA compatibility and strict trainable-state
load path; legacy full checkpoint and endpoint interpolation paths remain
unchanged.

## Files changed this session

- `pose_controlnet/release_artifact.py`
- `scripts/materialize_final_release.py`
- `inference.py`
- `tests/test_release_artifact.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages" /usr/bin/python3 -m py_compile pose_controlnet/release_artifact.py scripts/materialize_final_release.py inference.py tests/test_release_artifact.py
PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages" /usr/bin/python3 -m unittest tests.test_release_artifact tests.test_inference -v
PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages" /usr/bin/python3 scripts/materialize_final_release.py --output /tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors
sha256sum /tmp/krea2-release-final.XYB3BA/krea2-pose-control-mix025.safetensors
```

The unit suite passed 18 tests, including key mismatch rejection, shape
mismatch rejection, exact alpha interpolation, non-model-state exclusion, and
byte-deterministic serialization. The materializer independently reloaded and
exactly compared every one of the 450 saved tensors. A direct CPU resolver
check confirmed `inference.resolve_pose_candidate` loads the staged artifact
as canonical `mix-025` with all 215,488,512 float32 parameters. `git diff
--check` passed.

## Next recommended action

From the writable GH200 host shell, publish the exact artifact (no HF upload):

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages" /usr/bin/python3 scripts/materialize_final_release.py
```

Then run `sha256sum` on the intended NFS artifact; it must equal
`6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
