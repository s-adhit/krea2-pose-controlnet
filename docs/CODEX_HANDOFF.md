# Phase 1 handoff

## Current objective

Gate E — real Krea-2 Raw pose-control path verification — is complete and
host-verified **PASS** on the NVIDIA GH200.

The complete path from the immutable PoseBridge dataset through paired
preprocessing, Qwen VAE encoding, persistent latent shards, Krea-2 Raw model
loading, ControlInput expansion, rank-64 LoRA injection, flow-matching
forward/backward, and one AdamW optimization step has now been verified.

The next bounded milestone is **Gate F: production training mechanics**.

Do not begin the 10-step training smoke, 100-step training run, systemd
production service, or 6000-step production run in the same implementation
session. Gate F should first implement and locally verify the production
training machinery, then stop and provide the exact bounded GH200 smoke
command.

---

# Decisions in force

## Base model

- Base model: `krea/Krea-2-Raw`.
- Canonical checkpoint:
  `/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`.
- The checkpoint is gated and the accepted official Krea artifact is used.
- The monolithic checkpoint size observed on the production host is
  `26,283,332,608` bytes.
- Strict checkpoint compatibility is mandatory.
- Missing keys, unexpected keys, and shape mismatches are not permitted.

## Control representation

- Conditioning is the rendered skeleton/control image already present in the
  PoseBridge dataset.
- RGB and control images are spatially aligned.
- Both use exactly the same resize/crop/bucket geometry.
- RGB and control are independently encoded through the same Qwen VAE.
- The clean control latent is never noised before being supplied to the model.
- Image and control latents are patchified with identical geometry.
- Image token width is 64.
- Control token width is 64.
- Conditioning is introduced by spatially aligned **channel concatenation**:
  `64 + 64 = 128`.
- Token count must not change as a consequence of conditioning.

## ControlInput initialization

The original Krea image input projection is expanded from the original
image-token width to the concatenated image+control width.

The image half is initialized as an exact copy of the pretrained Krea input
projection.

The control half is initialized to exactly zero.

Therefore, at initialization:

- real-control and zero-control inference must be exactly identical;
- the pretrained image path is preserved;
- control initially has zero functional effect;
- the control half can nevertheless receive a nonzero gradient on the first
  backward pass.

This zero-impact initialization invariant has now been verified on the real
GH200 model.

## LoRA

LoRA rank is fixed at:

`64`

LoRA alpha is:

`64`

LoRA is applied to exactly the following eight linear paths in every one of
the 28 main transformer blocks:

- `attn.wq`
- `attn.wk`
- `attn.wv`
- `attn.wo`
- `attn.gate`
- `mlp.gate`
- `mlp.up`
- `mlp.down`

Total expected LoRA target modules:

`28 × 8 = 224`

Do not silently broaden or narrow the LoRA target set.

Standard zero-impact LoRA initialization is used. Because LoRA B starts at
zero, the first backward pass is expected to produce:

- LoRA A gradient = exactly zero for the representative module;
- LoRA B gradient = finite and nonzero.

After one optimizer step, a second backward must demonstrate finite nonzero
gradients for both A and B.

This behavior has now been verified on the real GH200 model.

## Precision

- Primary training compute precision: BF16.
- Qwen VAE encoding: BF16.
- Serialized latent shards: float32 CPU tensors.
- Do not change the project to FP16.
- Do not introduce quantized training without a separate design decision.

## Training objective

Training uses flow matching only.

For clean image latent `x_0`, sampled noise `noise`, and timestep `t`:

`x_t = t * noise + (1 - t) * x_0`

Target:

`noise - x_0`

Loss:

mean-squared error between the model prediction and the flow target.

No auxiliary pose loss, perceptual loss, reconstruction loss, or other
objective is part of the current design.

## Timestep sampling

The diagnostic and intended training design use logistic-normal timestep
sampling plus the configured resolution-dependent timestep shift.

Do not replace this with uniform timestep sampling without an explicit design
change.

## Optimizer

Optimizer family is **AdamW only**.

The Gate E diagnostic used:

- optimizer: `torch.optim.AdamW`
- learning rate: `1e-4`
- betas: `(0.9, 0.99)`
- weight decay: `0.0`

Gate F must preserve AdamW as the optimizer family.

Do not introduce:

- Adam
- Adafactor
- Lion
- 8-bit Adam
- bitsandbytes optimizers
- another optimizer family

without an explicit separate decision.

## Dataset membership

Immutable manifests define dataset membership.

Hugging Face storage layout does not define train/validation membership.

Source dataset files and immutable manifests are read-only.

Physical path resolution must continue to go exclusively through the
project-owned dataset indexing code.

---

# Verified host environment

Production host verification:

- OS architecture: Linux `aarch64`
- Python: `3.10.12`
- `uv`: `0.12.5`
- PyTorch: `2.7.0`
- PyTorch CUDA build: `12.8`
- torchvision: `0.22.0`
- Triton: `3.3.0`
- cuDNN: `90800`
- CUDA available: true
- BF16 CUDA support: true
- GPU: `NVIDIA GH200 480GB`

The project environment intentionally inherits the Lambda image's validated
system accelerator stack.

Torch, torchvision, CUDA, cuDNN, Triton, and NVIDIA packages are host-owned
and must not be independently replaced by project dependency resolution.

The project venv is created with system site packages enabled.

Do not use `pip install` to repair the project environment.

---

# Dependency/environment history

The project uses `uv`.

Torch-family accelerator packages are deliberately not project-managed.

The project encountered and resolved a NumPy ABI incompatibility caused by
NumPy 2.x interacting with host-compiled scientific/Torch packages.

The working project NumPy is:

`1.26.4`

The environment has also been verified with a compatible SciPy installation.

A later accidental environment state introduced a project-local incompatible
Torch build and broke torchvision. The environment was recreated correctly
with system site packages, restoring:

- torch `2.7.0` from `/usr/lib/python3/dist-packages`
- torchvision `0.22.0`
- Triton `3.3.0`
- CUDA `12.8`

Do not allow `uv` to replace the host-owned accelerator stack.

---

# Canonical persistent storage

Lambda NFS mount:

`/lambda/nfs/adhit`

Project persistent root:

`/lambda/nfs/adhit/krea2-pose`

Canonical immutable PoseBridge dataset snapshot:

`/lambda/nfs/adhit/krea2-pose/posebridge_hf`

Canonical persistent latent dataset:

`/lambda/nfs/adhit/krea2-pose/posebridge_latents`

Canonical Krea-2 Raw model location:

`/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`

The NFS filesystem is persistent and should be preferred for expensive
artifacts that should survive replacement of the GH200 instance.

---

# Gate A — GH200 environment

## Status

**PASS**

Verified on the real host:

- torch `2.7.0`
- torchvision `0.22.0`
- Triton `3.3.0`
- CUDA runtime `12.8`
- CUDA available
- GH200 visible
- BF16 supported

The BF16 SDPA/GQA smoke test also passed:

- Q heads: 48
- KV heads: 12
- head dimension: 128
- finite output
- BF16 output

No project-managed Torch/Triton/NVIDIA package is required.

---

# Gate B — physical dataset resolution

## Status

**PASS**

Canonical snapshot:

`/lambda/nfs/adhit/krea2-pose/posebridge_hf`

Physical structure contains:

- `images/shard_00` through `images/shard_08`
- `conditioning_images/shard_00` through
  `conditioning_images/shard_08`
- `metadata.jsonl`
- `manifests/train.jsonl`
- `manifests/val.jsonl`
- `manifests/diagnostic_val.jsonl`

Verified physical counts:

- RGB images: `17,495`
- control images: `17,495`
- non-cache payload files: `34,995`

RGB/control stem sets match exactly.

Manifest counts:

- train: `16,503`
- val: `889`
- diagnostic_val: `24`
- total manifest records: `17,416`

The splits are disjoint.

Every manifest record resolves to exactly one RGB/control pair.

Every caption is non-empty.

The shared project-owned resolver is:

`pose_controlnet.dataset_index`

It must remain the only physical path-resolution implementation used by
preprocessing, verification, training, and evaluation.

The index rejects:

- duplicate stems
- missing RGB/control counterparts
- malformed manifest filenames
- unresolved records
- duplicate split records
- overlapping split membership
- empty captions

---

# Gate C1 — paired geometric preprocessing

## Status

**PASS**

Shared paired geometry lives in:

`pose_controlnet.paired_preprocessing`

Fixed Krea buckets:

- `1024 × 1024`
- `896 × 1152`
- `1152 × 896`
- `832 × 1216`
- `1216 × 832`
- `768 × 1344`
- `1344 × 768`
- `704 × 1472`
- `1472 × 704`

Bucket selection uses nearest aspect ratio in log space.

Resize uses resize-to-cover with `round`.

Crop uses deterministic floor-offset center cropping.

RGB and control must have identical source dimensions before transformation.

Both members of a pair receive exactly the same:

- selected bucket
- resize scale
- resized dimensions
- crop coordinates

Path resolution and split membership are not duplicated in preprocessing.

---

# Gate C2 — real Qwen VAE integration

## Status

**PASS**

VAE:

`diffusers.AutoencoderKLQwenImage`

Source:

`Qwen/Qwen-Image`

Subfolder:

`vae`

Only the VAE artifact is required for preprocessing.

Input RGB/control images are normalized from `[0,255]` to `[-1,1]`.

Qwen's one-frame video input layout is used:

`B × 3 × 1 × H × W`

Encoding uses:

`latent_dist.sample()`

Raw Qwen latent shape:

`B × 16 × 1 × H/8 × W/8`

Normalization is:

`(z - vae.config.latents_mean) / vae.config.latents_std`

per latent channel.

Batch/time axes are then removed for the downstream stored layout:

`16 × H/8 × W/8`

Real GH200 smoke verified landscape, portrait, and square samples.

All tested RGB/control latent pairs:

- had matching shapes
- were finite
- had nonzero control statistics
- were encoded in BF16

Representative real smoke results included:

### Landscape

Bucket:

`1216 × 832`

Latent shape:

`16 × 104 × 152`

RGB RMS:

`0.5127766728401184`

Control RMS:

`0.7648327350616455`

### Portrait

Bucket:

`896 × 1152`

Latent shape:

`16 × 144 × 112`

RGB RMS:

`0.5327755808830261`

Control RMS:

`0.7697426080703735`

### Square

Bucket:

`1024 × 1024`

Latent shape:

`16 × 128 × 128`

RGB RMS:

`0.5577024817466736`

Control RMS:

`0.7617418169975281`

Gate C2 therefore verifies the real Qwen VAE path on the GH200.

---

# Gate D — persistent latent shards

## Status

**PASS**

Persistent latent root:

`/lambda/nfs/adhit/krea2-pose/posebridge_latents`

Shards are Torch `.pt` archives.

Each sample contains:

- float32 CPU image latent
- float32 CPU clean control latent
- caption
- stem/file identity
- split
- bucket metadata
- paired preprocessing geometry metadata

Default shard size:

`256 samples`

Typical paired latent payload is approximately 2 MiB, making a typical
256-sample shard roughly 0.5 GiB.

This provides reasonable NFS sequential throughput and recovery granularity.

## Atomicity and resume behavior

Shard creation:

1. writes to a same-directory temporary file;
2. flushes/fsyncs;
3. loads and hard-validates the result;
4. atomically renames it to its final shard path.

Existing final shards are reused only when they validate against the expected
deterministic split/stem range.

Temporary files are never considered complete.

`shards.json` is not proof of completion by itself.

A run records:

`complete=false`

before model loading/shard generation.

`complete=true`

is written only after full deterministic physical verification succeeds.

## Interrupted/resume smoke

A disposable GH200 smoke generated two samples per split.

After interruption:

`shards.json` correctly remained:

`complete=false`

The restart reused all six valid existing shards:

- train 2/2 reused
- val 2/2 reused
- diagnostic_val 2/2 reused

Partial verification passed:

- train: `2`
- val: `2`
- diagnostic_val: `2`
- total: `6`

This proved restart/reuse behavior.

## Full persistent shard verification

Full hard verification on the persistent NFS dataset passed:

- train: `16,503`
- val: `889`
- diagnostic_val: `24`
- total: `17,416`

Therefore Gate D is complete.

---

# W&B and local telemetry

## Status

**PASS**

W&B credentials are configured on the GH200 host.

Remote project:

- entity: `adhit-projects`
- project: `Krea-2-PoseControl-Lora`

A real remote connectivity run successfully synced.

Project telemetry implementation:

`pose_controlnet.wandb_logging.TrainingTelemetry`

Telemetry is deliberately failure-isolated.

Failures in:

- W&B import
- W&B initialization
- network communication
- metric logging
- image logging
- W&B finish

must not crash training.

Local JSONL telemetry is attempted independently.

Default local metrics path:

`runs/metrics.jsonl`

Runtime overrides include:

- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `WANDB_MODE`
- `WANDB_DISABLED`

Configuration fields whose names imply credentials are excluded from W&B
configuration serialization.

The project telemetry code does not read, persist, or write the W&B API key.

Future training should instantiate one `TrainingTelemetry` object and reuse its
named interfaces.

Do not implement a second independent telemetry system.

---

# Gate E — real Krea-2 Raw control path

## Status

**PASS**

The corrected diagnostic was executed successfully on the real NVIDIA GH200
using:

`/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`

and the verified persistent PoseBridge latent dataset.

The final diagnostic JSON reported:

`"status": "PASS"`

---

# Gate E checkpoint verification

Checkpoint size:

`26,283,332,608 bytes`

Checkpoint tensor count:

`430`

Strict compatibility:

- missing keys: `0`
- unexpected keys: `0`
- shape mismatches: `0`

Checkpoint configuration:

- channels: `16`
- features: `6144`
- heads: `48`
- KV heads: `12`
- layers: `28`
- multiplier: `4`
- patch: `2`
- timestep dimension: `256`
- theta: `1000.0`
- text dimension: `2560`
- text heads: `20`
- text KV heads: `20`
- text layers: `12`
- bias: `false`

---

# Gate E architecture verification

Verified architecture:

- hidden features: `6144`
- blocks: `28`
- attention heads: `48`
- attention KV heads: `12`
- attention head dimension: `128`
- MLP hidden features: `16,384`
- text input layers: `12`
- text layerwise blocks: `2`
- text refiner blocks: `2`
- text attention heads: `20`
- text attention KV heads: `20`

LoRA:

- rank: `64`
- target modules: `224`

Trainability audit:

- trainable parameters: `215,488,512`
- frozen parameters: `12,819,673,676`

Only the intended ControlInput and LoRA parameters are trainable.

Frozen backbone gradients were absent during the real backward diagnostic.

---

# Gate E real persistent sample

Sample:

`coco_100098_193288`

Source:

`train/train-00000.pt`

Sample number:

`0`

Bucket:

`1216 × 832`

Image latent shape:

`[1, 16, 104, 152]`

Control latent shape:

`[1, 16, 104, 152]`

Image RMS:

`0.5127767324447632`

Image standard deviation:

`0.5112274289131165`

Control RMS:

`0.7648329138755798`

Control standard deviation:

`0.7523605823516846`

Caption:

non-empty

The image/control latents are finite, aligned, and nonzero.

---

# Gate E ControlInput verification

Image token shape:

`[1, 3952, 64]`

Control token shape:

`[1, 3952, 64]`

Concatenated shape:

`[1, 3952, 128]`

ControlInput output shape:

`[1, 3952, 6144]`

Token count remained unchanged.

Pretrained image half exact match:

`true`

Image-half weight norm:

`34.62314987182617`

Control-half weight norm before backward:

`0.0`

This verifies the intended zero-initialized control expansion.

---

# Gate E diagnostic methodology correction

The first GH200 diagnostic attempt produced:

- initial max absolute real-vs-zero difference: `0.0625`
- initial RMS difference: `0.00848808791488409`

despite the control-half weights being exactly zero.

This was determined to be a diagnostic methodology problem rather than a
ControlInput architecture failure.

The old diagnostic compared:

- a train-mode, autograd-enabled, gradient-checkpointed real-control forward;
- retained through a backward pass;

against:

- a no-grad, non-gradient-checkpointed zero-control forward.

Those were not equivalent execution conditions.

The corrected diagnostic separates functional invariance checks from
gradient-bearing training checks.

Before any backward:

1. `model.eval()` is used;
2. one `torch.no_grad()` context is used;
3. `grad_ckpt=False` is used for both forwards;
4. the same precomputed image/context/timestep/position/mask inputs are used;
5. real-control and zero-control outputs are compared.

The corrected real GH200 result was:

- max absolute difference: `0.0`
- RMS difference: `0.0`

No tolerance was required.

This proves exact step-zero functional invariance.

The gradient-bearing path then separately restores:

`model.train()`

and performs the real training forward/backward.

After the AdamW step, the functional real/zero comparison is again performed
under equivalent eval/no-grad/non-checkpointed conditions.

---

# Gate E first backward

Flow timestep:

`0.7474455237388611`

Loss before optimizer step:

`0.3224196434020996`

ControlInput full gradient norm:

`4.091736316680908`

Control-half gradient norm:

`1.776748538017273`

The control half therefore receives a strong finite nonzero gradient despite
starting at exactly zero.

Representative LoRA module:

`blocks.0.attn.wq`

First-backward LoRA A gradient:

`0.0`

This is expected because LoRA B begins at zero.

First-backward LoRA B gradient:

`0.0021654702723026276`

This is finite and nonzero.

Frozen backbone gradients absent:

`true`

---

# Gate E optimizer step

Exactly one optimizer step was taken.

Optimizer:

`AdamW`

Diagnostic optimizer configuration:

- learning rate: `1e-4`
- betas: `(0.9, 0.99)`
- weight decay: `0.0`

Optimizer steps:

`1`

Control-half weight norm after the one step:

`0.06269358843564987`

The control projection therefore moved away from its exact zero
initialization.

---

# Gate E post-step control sensitivity

After the single AdamW step, the corrected functional comparison produced:

Real-vs-zero control max absolute difference:

`0.5078125`

Real-vs-zero control RMS difference:

`0.05853813514113426`

Both are finite and strongly nonzero.

This proves that after optimization the model output depends on the skeleton
control input.

---

# Gate E post-step LoRA gradient proof

After the one optimizer step, a separate second backward was performed without
taking a second optimizer step.

Representative LoRA A gradient:

`0.0001314304827246815`

Representative LoRA B gradient:

`0.0038118085358291864`

Both are finite and nonzero.

Loss after the one optimizer step:

`0.3022805154323578`

For this single deterministic diagnostic sample:

- loss before step: `0.3224196434020996`
- loss after step: `0.3022805154323578`

This one-step reduction is diagnostic only and must not be interpreted as a
training curve.

---

# Gate E memory observation

Peak CUDA allocated memory during the real diagnostic:

`31,220,089,856 bytes`

Approximately:

`31.22 GB`

This is an observed Gate E diagnostic allocation, not yet a production
training-memory benchmark.

Do not extrapolate production batch size solely from this number.

---

# Gate E conclusion

Gate E is **PASS**.

The real GH200 diagnostic proves all of the following simultaneously:

1. The official Krea-2 Raw checkpoint strictly matches the project model.
2. The intended Krea architecture is instantiated correctly.
3. Rank-64 LoRA is applied to exactly the intended 224 modules.
4. The pretrained image input projection is preserved exactly.
5. The new control projection starts exactly at zero.
6. Real control has exactly zero functional effect at initialization.
7. The control projection receives a finite nonzero gradient on the first
   backward.
8. LoRA B receives a finite nonzero first-backward gradient.
9. LoRA A's initial zero gradient matches standard zero-B LoRA initialization.
10. Exactly one AdamW step moves the control projection away from zero.
11. Real and zero control produce finite nonzero output divergence after the
    optimizer step.
12. A subsequent backward produces finite nonzero gradients in both LoRA A and
    LoRA B.
13. Frozen backbone parameters do not receive gradients.

The skeleton conditioning path is therefore functionally connected to the
training objective and capable of learning.

Do not reopen the Gate E architecture unless a later concrete regression
demonstrates a failure.

---

# Current overall gate status

- Gate A — GH200 environment: **PASS**
- Gate B — physical dataset resolution: **PASS**
- Gate C1 — paired geometry: **PASS**
- Gate C2 — real Qwen VAE integration: **PASS**
- Gate D — persistent full latent shards: **PASS**
- W&B/local telemetry: **PASS**
- Gate E — real Krea-2 control path: **PASS**
- Gate F — production training mechanics: **NEXT**

---

# Important project invariants

The following must not be changed casually during Gate F:

- Krea-2 Raw base
- official strict raw checkpoint
- rendered skeleton conditioning
- clean control latent
- spatial channel concatenation
- exact zero control-half initialization
- rank-64 LoRA
- exactly 224 LoRA target modules
- BF16 training compute
- float32 serialized latent shards
- flow-matching MSE only
- logistic-normal timestep sampling/resolution shift
- AdamW optimizer family
- immutable manifest-defined splits
- existing paired preprocessing implementation
- existing Qwen VAE implementation
- existing persistent latent format
- existing `TrainingTelemetry`
- frozen pretrained backbone outside the explicitly trainable parameters

Do not duplicate project-owned implementations for dataset indexing,
preprocessing, VAE encoding, telemetry, or model/control construction.

---

# Gate F — next bounded milestone

## Objective

Implement production-safe training mechanics around the already verified
model, data, control, loss, and telemetry components.

This milestone is about trainer engineering.

It is not an architecture redesign.

It must not modify the verified pose-control mechanism simply to make the
trainer easier to implement.

## Gate F implementation scope

Gate F should implement the production training entry point and required
training mechanics, including:

- real latent-shard loading
- deterministic training sample/shard ordering
- resumable data position where required by the existing checkpoint design
- Krea-2 Raw strict loading
- existing rank-64 pose-control model construction
- BF16 forward/backward
- existing flow-matching objective
- AdamW optimizer
- learning-rate scheduling
- configured warmup
- gradient accumulation
- effective batch-size accounting
- gradient clipping
- optimizer-step accounting
- validation flow loss
- CUDA memory telemetry
- throughput telemetry
- existing W&B/local `TrainingTelemetry`
- checkpoint metadata/status integration
- controlled shutdown/cleanup

The existing project components must be reused rather than independently
reimplemented.

## Training batch target

The current intended effective training batch is:

`32`

Gate F should implement gradient accumulation as necessary to reach the
configured effective batch without assuming that the entire effective batch
fits into one GH200 forward/backward.

The actual safe microbatch must be established by the later bounded GH200
smoke rather than guessed from Gate E memory usage.

## Warmup

Current intended warmup:

`200 optimizer steps`

Scheduler/warmup implementation should operate on optimizer steps, not raw
microbatch forwards.

## Gradient clipping

Gate F must include gradient clipping and telemetry for the resulting gradient
norm.

Do not silently omit clipping.

## Validation

Validation must use the immutable validation split and the same project-owned
flow formulation.

Validation should report flow loss without performing optimizer updates.

Diagnostic image generation, where already supported by project telemetry,
should remain sparse and must not become part of every training step.

---

# Gate F exclusions

The Gate F implementation session must **not**:

- start the production 6000-step run
- start a 100-step run
- start a 10-step GH200 run unless explicitly authorized afterward
- redesign the model
- change LoRA rank
- change LoRA target modules
- change control representation
- change the Qwen VAE
- regenerate latent shards
- modify immutable manifests
- change optimizer family
- introduce 8-bit optimization
- install a new Torch/CUDA stack
- add a second telemetry implementation
- add unrelated infrastructure
- commit or push unless separately requested

Implement, test, document, and stop.

---

# Expected Gate F local verification

At minimum, Gate F should provide focused tests for the newly introduced
training mechanics.

Tests should cover relevant invariants such as:

- optimizer is AdamW
- only intended parameters enter the optimizer
- effective batch/gradient accumulation accounting
- optimizer step increments only after the intended accumulation
- scheduler advances on optimizer steps
- warmup is expressed in optimizer steps
- gradient clipping occurs at the correct point
- deterministic sample ordering
- validation performs no optimizer update
- checkpoint/resume restores required training state
- telemetry failures remain nonfatal
- frozen parameters remain excluded from optimization

Do not substitute huge repo-wide tests for focused tests unless necessary.

Run syntax/compile checks for modified training files.

Run:

`git diff --check`

and:

`git status --short`

before stopping.

---

# Current blockers

None for Gates A through E.

Gate F has not yet been implemented/verified.

No production training should begin until Gate F implementation and its
bounded GH200 smoke are separately reviewed.

---

# Exact next recommended action

Start a new bounded Codex session for Gate F implementation.

Suggested command:

    cd ~/Krea-2-Pose-ControlNet

    codex exec \
      -m gpt-5.6-terra \
      -c model_reasoning_effort=medium \
      "
      Read AGENTS.md and docs/CODEX_HANDOFF.md.

      Gate E is host-verified PASS.

      Do only the exact next bounded milestone: Gate F production training
      mechanics.

      Reuse the existing dataset/shard, paired preprocessing, model/control,
      flow objective, TrainingTelemetry, and checkpoint interfaces.

      Required:
      - production training entry point
      - deterministic/resumable latent-shard iteration
      - strict Krea-2 Raw loading
      - existing rank-64 control/LoRA construction
      - BF16 training
      - flow-matching MSE only
      - AdamW only
      - scheduler with 200 optimizer-step warmup
      - gradient accumulation for configured effective batch 32
      - gradient clipping
      - optimizer-step accounting
      - validation flow loss
      - existing W&B/local telemetry integration
      - checkpoint/resume state needed for deterministic continuation
      - controlled shutdown

      Do not change architecture, LoRA rank/targets, control representation,
      VAE, latent format, dataset manifests, optimizer family, Torch/CUDA, or
      telemetry design.

      Do not start a 10-step, 100-step, systemd, or 6000-step training run.

      Add focused tests for the new mechanics.
      Run targeted tests and py_compile.
      Run git diff --check.
      Run git status --short.

      Rewrite docs/CODEX_HANDOFF.md with:
      - exactly what Gate F implemented
      - tests/results
      - unresolved issues
      - exact bounded real-GH200 smoke command

      STOP after implementation/testing/documentation.
      Do not commit or push.
      "

After Gate F implementation is reviewed, run only the separately authorized
bounded GH200 training smoke.

Do not jump directly from Gate E PASS to the full production run.
