# Project handoff

## Current status

The bounded manifest-identity preflight fix for the full 16,503-sample 768
cache builder is complete. No cache, pose sidecar, training, generation,
evaluation, or throughput benchmark was run this session.

## Full 768 manifest identity contract

- Authoritative project manifest:
  `data/manifests/train.jsonl`.
- Snapshot manifest:
  `/lambda/nfs/adhit/krea2-pose/posebridge_hf/manifests/train.jsonl`.
- Both must parse to exactly 16,503 JSON-object records.
- The complete parsed records (including `conditioning_image`) must match
  exactly and in order. This is the scientific manifest identity.
- The ordered `file_name` values from the snapshot must exactly match the
  resolved snapshot `ManifestRecord` order; the corresponding unique ordered
  stem list is separately hashed and persisted.
- `manifest_records_sha256` is the canonical, key-order-independent digest of
  the ordered parsed records. It and `ordered_stems_sha256` bind cache and
  sidecar metadata.
- Raw SHA-256 values for the project and snapshot files are recorded only as
  separate provenance fields (`authoritative_train_manifest_raw_sha256` and
  `snapshot_train_manifest_raw_sha256`). They may differ when JSON formatting,
  key ordering, or line endings differ; they are not used as scientific
  identity or cache-conflict keys.
- Any changed record content, changed stem, changed row order, malformed row,
  count mismatch, or disagreement with resolved snapshot order fails closed.

## Root cause and correction

`_identity` compared raw project JSON objects against dictionaries rebuilt
from `ManifestRecord`. The resolver intentionally retains only `file_name` and
`text`, while both legitimate manifests also contain `conditioning_image`.
Thus every legitimate three-field raw record differed from a reconstructed
two-field record even though the project and snapshot manifests parsed to the
same ordered records.

`pose_controlnet/full_768_cache.py` now parses both manifest files directly
and compares their complete ordered JSON records before any cache root or VAE
work. It retains raw hashes as provenance and uses a canonical parsed-record
digest for artifact identity.

## Files changed this session

- `pose_controlnet/full_768_cache.py`
- `tests/test_full_768_cache.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from pose_controlnet.dataset_index import validate_posebridge_snapshot
from pose_controlnet.full_768_cache import _identity
root = Path('/lambda/nfs/adhit/krea2-pose/posebridge_hf')
snapshot = validate_posebridge_snapshot(root)
print(_identity(snapshot.records_by_split['train'],
                Path('data/manifests/train.jsonl').resolve(),
                root / 'manifests/train.jsonl')['sample_count'])
PY
```

This printed `16503`; the parsed-record and ordered-stem digests were stable,
while the two raw SHA values differed as expected.

PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_full_768_cache tests.test_dataset_index tests.test_shards -v
PYTHONPATH=. python -m py_compile pose_controlnet/full_768_cache.py scripts/build_full_768_cache.py scripts/verify_full_768_cache.py tests/test_full_768_cache.py
```

The focused identity tests prove acceptance of formatting-only raw differences
without VAE loading, and rejection of reordering, changed stems, changed
`conditioning_image`, and conflicting resolved order.

## Exact safe retry command

```bash
cd /home/ubuntu/krea2-pose-controlnet
export DATASET=/lambda/nfs/adhit/krea2-pose/posebridge_hf
export TRAIN_MANIFEST=/home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl
export LATENT_768=/lambda/nfs/adhit/krea2-pose/posebridge_latents_768
export POSE_SOURCE=/lambda/nfs/adhit/krea2-pose/pose_targets_v3
export POSE_768=/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768
PYTHONPATH=. python scripts/build_full_768_cache.py \
  --dataset-root "$DATASET" --output-root "$LATENT_768" \
  --train-manifest "$TRAIN_MANIFEST" \
  --pose-source "$POSE_SOURCE" --pose-output "$POSE_768" --device cuda
```

This command has not been run in this session. Do not start the throughput
matrix until its cache verifier passes.
