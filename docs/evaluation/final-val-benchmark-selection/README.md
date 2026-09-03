# Final-val Turbo evaluation

The immutable 48-item benchmark is `final_val_benchmark_spec.json`. The
historical 24-item diagnostic Turbo benchmark is separate and unchanged.

Only two real controlled checkpoints are currently allowed:

- `parent-4000` — step 4000 from the production cooldown parent.
- `finish-control-a4300` — step 4300 from the finish-control branch.

Turbo base/zero-adapter evaluation and checkpoint interpolation are intentionally
not supported by this entry point. Every action rechecks the spec digest, the
48 cached latent/control/text identities and per-stem seeds, DatasetIndex
control resolution, the exact selected checkpoint step, and the locked Turbo
settings: 8 steps, CFG 0, mu 1.15, control scale 1.0.

Run each action from the GH200 host shell. Choose a distinct output directory
for each candidate; generation never overwrites a partial or conflicting set.

```bash
uv run python scripts/final_val_turbo_benchmark.py preflight \
  --candidate parent-4000 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000

uv run python scripts/final_val_turbo_benchmark.py generate \
  --candidate parent-4000 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000

uv run python scripts/final_val_turbo_benchmark.py preflight \
  --candidate finish-control-a4300 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300

uv run python scripts/final_val_turbo_benchmark.py generate \
  --candidate finish-control-a4300 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

Scoring requires a separately prepared immutable authoritative pose sidecar for
these exact 48 stems in frozen order. It must not be the diagnostic sidecar.
After generation, score and report each candidate with the same output root:

```bash
uv run python scripts/final_val_turbo_benchmark.py score \
  --candidate parent-4000 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000 \
  --reference-sidecar /path/to/final_val_pose_sidecar

uv run python scripts/final_val_turbo_benchmark.py report \
  --candidate parent-4000 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000

uv run python scripts/final_val_turbo_benchmark.py score \
  --candidate finish-control-a4300 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300 \
  --reference-sidecar /path/to/final_val_pose_sidecar

uv run python scripts/final_val_turbo_benchmark.py report \
  --candidate finish-control-a4300 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

`--help` documents cache, DatasetIndex, Turbo checkpoint, and CLIP overrides;
only the exact candidate checkpoint and step can be evaluated.
