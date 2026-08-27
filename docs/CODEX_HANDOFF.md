# Phase 1 handoff

## Current bounded objective and status

The timestep-exposure-only continuation is implemented and validated, but has
**not** been launched. No training/resume, optimizer update, checkpoint write,
HF upload, W&B run, commit, or push occurred in this session.

The prior completed LR=5e-5 branch remains immutable:

- Run/HF namespace: `pose-learning-900-lr5e5-to1500/full/` in
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`.
- Verified exact local source:
  `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`.
- Embedded state: `global_step=1500`, `epoch=2`, `batch_position=7502`,
  `lr=5e-5`, scheduler `step_count=1500`, base LR `5e-5`, warmup `200`.
- Completed Turbo evaluation support remains in
  `scripts/turbo_lr5e5_benchmark.py`; it is read-only and uses unchanged
  Turbo 8-step/CFG-0 sampling. Its reported step-1500 results are CLIP
  `0.336843`, detection `0.923077`, joint coverage `0.927184`, PCK@0.05
  `0.054612`, PCK@0.10 `0.180825`, PCK@0.20 `0.412621`.

## Timestep continuation design

New isolated run: `pose-learning-1500-timestep-lowmid20-to1800`.

The executable original training sampler is per-example:

```text
z = torch.randn(batch_size, generator=dedicated CUDA flow generator)
u = sigmoid(z)
mu = ((1.15 - 0.5) / (6400 - 256)) * seq_len + (0.5 - slope * 256)
t = exp(mu) * u / (exp(mu) * u + 1 - u)
```

It is called by `train._flow_loss` before noise, with
`seq_len=(latent_h/patch)*(latent_w/patch)` and `patch=2`. It uses one shared
CUDA `torch.Generator` per process. Its state is saved as
`flow_generator_state` and restored after CPU/Python/NumPy/CUDA RNG state;
sampling is per example, not per batch. Actual train buckets have
`seq_len=3952..4096`, hence `mu=0.891015625..0.90625`.

The proposed sampler leaves the disabled path exactly unchanged. When enabled:

```text
normal = sigmoid(torch.randn(...))       # exact original pre-shift branch
mask = torch.rand(...) < 0.20
aux = Uniform(0.04359494981207863, 0.3773562340267345)
u = where(mask, aux, normal)
t = same existing shift_timestep(u, mu) # no post-shift clamp
```

The fixed auxiliary bounds are the inverse existing shift for final `t=0.1`
at the lowest actual `mu`, and final `t=0.6` at the highest actual `mu`.
Thus the auxiliary component maps to approximately final `t=0.1..0.6` for
every actual bucket without changing shift math. New checkpoint config records
`timestep_aux_prob/min/max`. The continuation config derives only from the
exact local source and rejects every non-timestep config delta. It preserves
LR, AdamW state, scheduler, warmup progress, progress counters, all RNG/data
state, model/data/precision settings, and HF/W&B settings.

## Read-only distribution audit

`scripts/audit_timestep_exposure.py --samples-per-bucket 100000` sampled
900,000 values (100,000 for each of the 9 actual buckets), weighted by all
16,503 train samples, with seed `420300`:

| sampler | mean | median | 0-.2 | .2-.4 | .4-.6 | .6-.8 | .8-1 | aux route |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | .678551 | .710412 | 1.0980% | 8.4629% | 21.5175% | 37.6456% | 31.2760% | 0.0000% |
| 80/20 | .618331 | .640524 | 3.8254% | 14.0587% | 27.1357% | 30.0595% | 24.9208% | 20.0500% |

## Files changed and tests

- `pose_controlnet/config.py`
- `pose_controlnet/diffusion.py`
- `train.py`
- `scripts/audit_timestep_exposure.py`
- `tests/test_timestep_exposure.py`
- `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile ...` for
the changed Python files.

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest discover -s tests -p 'test_*.py'` — 144 tests.

## Exact future GH200 launch and immediate verification

Do not run this without explicit approval. The special selector accepts no
mutable source/target arguments and reads only the exact source checkpoint:

```bash
cd /home/ubuntu/Krea-2-Pose-ControlNet
tmux new-session -d -s pose-learning-1500-timestep-lowmid20-to1800 \
  'cd /home/ubuntu/Krea-2-Pose-ControlNet && export UV_CACHE_DIR=/tmp/krea_uv_cache && exec uv run python train.py --timestep-lowmid-1500-to1800'
```

Target local root:
`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800`.
HF target: `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`, namespace
`pose-learning-1500-timestep-lowmid20-to1800/full/`. W&B run name is
`pose-learning-1500-timestep-lowmid20-to1800` in project
`Krea-2-PoseControl-Lora` / entity `adhit-projects`.

The preserved `save_every=25` and `hf_mirror_every_steps=100` guarantee saved
and step-mirrored checkpoints at 1600, 1700, and 1800. Immediately inspect:

```bash
tmux capture-pane -pt pose-learning-1500-timestep-lowmid20-to1800 -S -200 \
  | rg '\[timestep-branch\]|effective_batch|runtime'
```

Expected log fields: exact source `step_001500.pt`, optimizer/global step
`1500`, LR `5e-5`, scheduler step `1500`, warmup `200`, aux probability `.2`,
pre-shift support `[0.04359494981207863, 0.3773562340267345)`, checkpoints
`(1600, 1700, 1800)`, HF target namespace, and W&B run name.

## Next action

Stop. Await explicit authorization before executing the GH200 launch command.
