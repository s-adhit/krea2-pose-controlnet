# Krea-2 Pose Control-LoRA — Agent Instructions

## Mission

This repository has one permanent objective:

**skeleton control image -> clean control latent -> channel-concat with noisy image latent -> expanded input projection -> Krea-2 Raw + rank-64 LoRA -> flow-matching target**

This is a single-scope project.

The only deliverable is a production-ready skeleton-conditioned Krea-2 Raw Pose Control-LoRA.

There is no Phase 2 or Phase 3 roadmap for this repository.

Do not implement, plan, scaffold, reserve interfaces for, preserve compatibility with, or spend context on:

- RGB-reference adapters;
-- direct RGB-to-pose latent adapters;
- RGB-to-control-latent adapters;
- Phase 2;
- Phase 3;
- later research phases built on this repository.

Do not add abstractions, config fields, interfaces, data paths, model hooks, or architectural complexity solely for hypothetical future RGB-reference work.

The reference depth-ControlNet repository is a **third-party implementation by Tanmay**, not Krea's official training recipe.

Use it only as a behavioral/architectural reference where necessary.

Our training, data, logging, evaluation, checkpoint, recovery, and production infrastructure must be project-owned.


## Non-negotiable decisions

- Base model for training: **Krea-2 Raw**.
- Control: rendered pose skeleton.
- Control mechanism: spatially aligned **channel concatenation**.
- Control is not represented as extra tokens.
- Control is not a classical side-branch ControlNet.
- Image latent is noisy at the sampled flow timestep.
- Control latent remains clean.
- LoRA rank: **64**.
- Production objective: **flow-matching MSE plus normalized-coordinate pose-consistency Huber**.
- Main production/control branch: `lambda_pose = 0.04`; the controlled
  timestep exposure contract is part of checkpoint provenance.
- PCK and related pose metrics are evaluation-only.
- Seed: **42**.
- Baseline optimizer unless explicitly changed by the user:
  - AdamW
  - learning rate `1e-4`
  - betas `(0.9, 0.99)`
  - weight decay `0.0`
- Warmup: 200 optimizer steps.
- Target effective batch: 32.
- Determine GH200 microbatch size only after measuring actual VRAM usage and throughput.
- Baseline maximum training steps: 6000.
- Caption dropout: 0.10.
- Precision: BF16.
- Use **uv**, not pip, for project dependency management.
- Do not replace Lambda's known-working CUDA/PyTorch/cuDNN/Triton stack unless a concrete incompatibility is demonstrated.
- This repository has no future RGB-reference roadmap. Do not preserve complexity for cancelled phases.


## Verified GH200 environment

Host-verified from the normal Lambda SSH shell:

- Linux ARM64 / aarch64.
- NVIDIA GH200.
- Approximately 96 GB GPU HBM available.
- Python 3.10.12.
- PyTorch 2.7.0.
- PyTorch CUDA runtime 12.8.
- cuDNN 9.8.
- Triton 3.3.0.
- BF16 support: PASS.
- SDPA test: PASS.
- `torch.compile`: PASS.
- uv 0.12.5.

The NVIDIA driver may report a newer CUDA capability than the CUDA runtime bundled with PyTorch. This alone is not a reason to change the environment.

The Codex execution/audit shell may not always expose the host GPU.

A failure of:

- `nvidia-smi`;
- `torch.cuda.is_available()`;
- BF16;
- SDPA;
- CUDA tests;

inside a Codex sandbox must not automatically override the successful host-shell verification above.

Do not upgrade or replace:

- torch;
- torchvision;
- CUDA;
- cuDNN;
- Triton;
- NVIDIA drivers/packages;

without a demonstrated compatibility problem and explicit approval.


## Data invariants

Treat source data and immutable manifests as read-only.

Expected clean split after known exclusions:

- train: 16,503
- validation: 889 (held out from training and used for inference benchmarking)
- diagnostic val: 24
- total used: 17,416

The larger source corpus contains approximately 17,495 RGB/control pairs before the known exclusions.

Raw source/control pairs share stems but use different extensions.

Immutable manifest schema:

- `file_name`: bare `<stem>.jpg`
- `text`: caption

Controls are matched by stem and use `.png`.

Example conceptual relationship:

```text
manifest:
abc123.jpg

resolved physical files:
<some HF prefix>/abc123.jpg
<some HF prefix>/abc123.png
```

The immutable manifest filename does **not** encode the physical Hugging Face storage path.

Terminology is deliberate: the diagnostic split is the development/selection
benchmark. The validation split is held out from training but is inspected for
inference benchmarking; it is not an untouched final test set.

The Hugging Face dataset may contain many path prefixes/directories because of how the approximately 35,000 files were uploaded.

Therefore:

- do not infer physical paths from manifest filenames;
- do not assume a flat `images/` directory;
- do not assume a flat `conditioning_images/` directory;
- do not use upload path prefixes as train/validation split definitions;
- immutable manifests define dataset membership;
- physical storage layout is only a storage concern.

After downloading the HF snapshot:

1. discover the real physical layout;
2. build a read-only `stem -> RGB physical path` index;
3. build a read-only `stem -> control physical path` index;
4. detect duplicate stems;
5. detect missing RGB/control counterparts;
6. validate all immutable manifest records against the physical index.

Prefer one shared dataset-index implementation used by preprocessing, training, verification, and evaluation rather than duplicating path-resolution logic.

Normalize basename/stem joins deliberately.

Never silently replace a missing caption with `""`.

RGB and skeleton control must undergo **the exact same resize/crop/bucket geometry**.

Source data, immutable manifests, and downloaded HF files must not be modified, renamed, reorganized, or duplicated unless a demonstrated technical requirement makes that necessary.


## Current state vs required production state

Anything in this file described as **required** is a target production state.

Do not assume it is already implemented.

Always inspect the actual repository before relying on a component.

Known areas that may require implementation or verification include:

- `train.py`;
- `prepare_shards.py`;
- `scripts/verify_shards.py`;
- dataset path resolution;
- paired preprocessing;
- VAE preprocessing;
- production flow-matching loss loop;
- optimizer and scheduler implementation;
- validation/evaluation;
- control-path diagnostics;
- W&B telemetry;
- local metrics fallback;
- full resumable checkpointing;
- hourly checkpoint creation;
- Hugging Face checkpoint mirroring;
- automatic resume;
- signal handling;
- unattended service execution;
- systemd restart behavior;
- kill/resume verification.

Do not let README claims or desired architecture documentation substitute for inspecting actual code and tests.


## Important reference behavior

The reference `ControlInputLayer` expands the input feature width while preserving spatial token count.

Conceptually:

```text
image latent tokens:   N x C
control latent tokens: N x C

channel concat:
N x 2C

expanded input projection:
N x hidden_width
```

This is channel concatenation, not token concatenation.

The transformer continues to operate on the same spatial token count `N`.

The reference initialization behavior:

- pretrained image-input weights are copied into the image half;
- the newly introduced control half is initialized to zero.

Therefore:

**At initialization, changing the skeleton may not change model output. This is expected.**

Do not incorrectly declare the control path broken because step-0 outputs are identical.

What must instead be verified:

1. control tensors are non-empty;
2. control tensors are finite;
3. control tensors have expected spatial/channel shapes;
4. control tensors are spatially aligned with image latents;
5. after the first backward pass, the gradient norm of the control half of `ControlInputLayer.weight` is finite and **> 0**;
6. representative LoRA tensors receive finite gradients;
7. frozen backbone tensors remain frozen;
8. after one or more optimizer steps, real-control and zero-control forward passes with otherwise identical inputs begin to diverge.


## Repository inspection discipline

Do not recursively dump the repository into context.

At the beginning of a fresh implementation area:

1. run:

```bash
git status --short
git log -5 --oneline
```

2. use targeted discovery:

```bash
rg --files
rg "<symbol>"
```

while avoiding irrelevant/generated directories.

Do not recursively scan:

- `.git/`
- `.venv/`
- `data/`
- `checkpoints/`
- `outputs/`
- `runs/`
- caches
- model weights
- latent shards
- image datasets
- generated artifacts

unless a specific task requires one of them.

Relevant project files may include:

- `train.py`
- `prepare_shards.py`
- `evaluate.py`
- `pose_controlnet/config.py`
- `pose_controlnet/data.py`
- `pose_controlnet/diffusion.py`
- `pose_controlnet/model.py`
- `pose_controlnet/text_encoder.py`
- `pose_controlnet/seed.py`
- `pose_controlnet/wandb_logging.py`
- `pose_controlnet/checkpointing.py`
- `scripts/check_environment.py`
- `scripts/verify_shards.py`
- smoke-test scripts
- `base_model/mmdit.py`
- `base_model/k2_lora.py`

Reference/vendor files such as:

- reference `k2_lora.py`
- reference `mmdit.py`
- `trainer/sampling.py`
- `trainer/train_control_lora.py`

should be read only when needed to verify architecture or training math.

Do not broadly refactor the reference architecture before testing the real checkpoint.


## Hands-off session operating protocol

Default behavior for every Codex session is **hands-off execution within the stated task boundary**.

Do not stop after every minor step to ask for permission.

Within the bounded task:

- inspect;
- implement;
- test;
- diagnose failures;
- repair failures;
- re-test;
- update the handoff;
- stop when the bounded milestone is complete.

Routine implementation decisions that do not alter the experiment do not require user approval.


### At session start

1. Read `AGENTS.md`.
2. Read `docs/CODEX_HANDOFF.md` if it exists.
3. Treat the handoff as the current operational state, but verify material claims against code/tests when relevant.
4. Run:

```bash
git status --short
```

5. Never overwrite unrelated user changes.
6. Determine the **single bounded objective** from the user prompt or exact next action recorded in the handoff.
7. Do not expand into another milestone unless required to make the current objective work.
8. Use targeted `rg`/`rg --files` searches.
9. Open only files needed for the current objective.


### During the session

- Work autonomously through normal implementation details.
- Package installation is allowed only when consistent with the current task and environment rules.
- Prefer the smallest coherent patch satisfying the acceptance criteria.
- Run the narrowest useful test after material changes.
- If a test fails, investigate and fix it rather than immediately returning the failure.
- Do not make unrelated refactors.
- Do not begin another major milestone after the assigned one is complete.
- Do not commit or push unless explicitly authorized for that session.
- Do not mutate immutable manifests.
- Do not mutate source datasets.
- Do not change the training objective without explicit approval.
- Do not change the model architecture merely for speculative future flexibility.
- Never treat temporary W&B/HF/network failures as reasons to stop implementation of robust retry/nonfatal behavior.
- Never change the working torch/CUDA/cuDNN/Triton/NVIDIA stack without demonstrated incompatibility and explicit approval.
- Never launch the paid 6000-step production run without explicit user approval.


### Escalate only for a genuine blocker

Stop and request user input only for:

- missing credentials or access;
- inaccessible required external artifacts;
- destructive or irreversible operations not already authorized;
- contradictory requirements;
- a material training-methodology decision;
- a model-architecture decision that changes the experiment;
- a dependency incompatibility that would require replacing the working GPU stack;
- production training launch authorization.

Do not escalate merely because:

- a normal unit test failed;
- an import needs a compatible dependency;
- a local implementation bug exists;
- a small refactor is needed;
- W&B is temporarily offline;
- HF is temporarily unavailable;
- a retryable network error occurred.


## Codex model / effort routing

Use the least expensive model capable of completing the task.

### Luna / low reasoning

Use for:

- package installation;
- version checks;
- simple shell operations;
- environment inspection;
- file discovery;
- basic config changes;
- formatting;
- mechanical validation;
- simple Git/status checks.

### Terra / medium reasoning

Use for:

- bounded Python implementation;
- dataset indexing;
- preprocessing;
- loaders;
- shard creation;
- shard verification;
- W&B integration;
- checkpoint plumbing;
- resume plumbing;
- service configuration;
- normal debugging;
- test implementation.

### Sol / high reasoning

Reserve for:

- architecture/math changes;
- control-path correctness;
- difficult numerical problems;
- difficult CUDA/runtime failures;
- optimizer/training-method decisions;
- subtle resume/determinism bugs;
- final production-readiness reviews;
- decisions that could waste substantial GH200 compute.

Do not use Sol merely to execute shell commands or mechanical edits.

Do not spawn multiple agents when one bounded agent/task can complete the work.


## Context and token discipline

Keep Codex sessions short and task-specific.

A fresh session should normally need only:

- `AGENTS.md`;
- `docs/CODEX_HANDOFF.md`;
- files directly relevant to the current task.

Rules:

- do not reread historical chat transcripts;
- do not read old handoff documents unless `CODEX_HANDOFF.md` explicitly points to a specific detail that is required;
- do not recursively scan large directories;
- search for symbols/paths before opening entire files;
- prefer reading slices around relevant definitions;
- do not repeatedly reread unchanged files;
- use `git diff` for recently changed content;
- redirect verbose command output to logs/files when useful;
- inspect relevant errors/tails instead of loading entire logs;
- do not start a second major milestone in the same session after completing the first;
- update the handoff and end the session at a natural milestone.

Preferred session lifecycle:

```text
read AGENTS + handoff
        ↓
single bounded objective
        ↓
targeted inspection
        ↓
implementation
        ↓
targeted tests
        ↓
fix/retest if needed
        ↓
update handoff
        ↓
git diff --check
git status --short
        ↓
stop
```


## Session completion — mandatory handoff

Before ending **every session that meaningfully inspects or changes project state**, update:

```text
docs/CODEX_HANDOFF.md
```

Create it if absent.

This is mandatory and should happen automatically.

The handoff is the single cross-session operational state file.

Do not create competing status/handoff documents unless explicitly requested.

Keep it concise and current.

Target size:

```text
roughly 2–5 KB
```

It should contain only:

- current project objective;
- verified environment facts;
- decisions currently in force;
- completed/green gates;
- current failures/blockers;
- files changed this session;
- exact tests/commands run with PASS/FAIL;
- important new findings;
- exact next recommended action.

Rewrite stale sections rather than appending an ever-growing chronology.

Do not paste:

- large logs;
- full diffs;
- stack traces;
- source files;
- chat transcripts;
- historical cancelled plans.

A fresh Codex session should be able to continue by reading only:

```text
AGENTS.md
docs/CODEX_HANDOFF.md
relevant source files
```

If code/config was changed, finish with:

1. targeted tests green or a clearly documented genuine blocker;
2. `git diff --check`;
3. `git status --short`;
4. concise handoff update.

Do not automatically commit or push unless the user explicitly authorizes Git writes for that session.

Never include in Git:

- credentials;
- tokens;
- private keys;
- model weights;
- datasets;
- latent caches;
- checkpoints;
- generated secrets.


## Phase 1 verification gates

Do not start the 6000-step production run until every gate is green.


### Gate A — GH200 environment

Verify:

- Linux ARM64.
- CUDA visible from the actual production shell/service environment.
- BF16 supported.
- PyTorch version recorded.
- CUDA runtime recorded.
- driver recorded.
- cuDNN version recorded.
- Triton version recorded.
- SDPA/attention smoke test passes.
- `torch.compile` passes if production intends to use it.
- uv environment works.
- creating the project environment does not accidentally replace the known-working system torch/CUDA stack.
- required disk/mount paths have sufficient free space.


### Gate B — dataset physical resolution

After the HF dataset is downloaded:

- inspect real path prefixes;
- count RGB files;
- count pose-control files;
- build unique stem indices;
- verify no duplicate ambiguous stems;
- verify every required RGB has its control;
- verify every required control resolves to an RGB;
- validate every immutable manifest row;
- confirm train/val/diagnostic membership remains manifest-defined;
- do not allow physical HF prefixes to influence split membership.


### Gate C — data preprocessing

Verify:

- manifest schema matches actual loader assumptions;
- train count = 16,503;
- representative val count = 889;
- diagnostic val count = 24;
- total used = 17,416;
- captions are non-empty;
- source/control stem alignment is correct;
- RGB and control receive identical bucket selection;
- RGB and control receive identical resize geometry;
- RGB and control receive identical crop geometry;
- VAE outputs are finite;
- image/control latent shapes align;
- control latent contains measurable nonzero signal;
- cached/prepared records preserve enough metadata for debugging/reproducibility.


### Gate D — model and checkpoint compatibility

Before training:

- load the real Krea-2 Raw checkpoint into the intended architecture;
- verify strict or intentionally documented state-dict loading;
- verify expanded `ControlInputLayer` behavior;
- verify expected model dimensions;
- verify rank-64 LoRA injection;
- verify expected target modules;
- assert frozen backbone parameters remain frozen;
- assert only intended control/LoRA parameters are trainable.


### Gate E — control-path proof

Add a short diagnostic mode/test that records:

- image latent RMS/std;
- control latent RMS/std;
- concatenated tensor shape;
- `ControlInputLayer` image-half weight norm;
- control-half weight norm;
- control-half gradient norm after backward;
- representative LoRA A/B gradient norms.

Use the same image/noise/text input with:

- real pose control;
- zero pose control.

Expected behavior:

Before training:

```text
real-control output ~= zero-control output
```

because the control-half projection is zero-initialized.

After backward:

```text
control-half gradient norm > 0
```

After optimizer update(s):

```text
real-control output != zero-control output
```

with a finite nonzero difference.

Fail loudly on:

- NaN/Inf;
- shape mismatch;
- empty controls;
- zero control-half gradient after backward;
- absent LoRA gradients;
- accidental frozen-backbone trainability.


### Gate F — training mechanics

Prove in this order:

1. one real batch loads;
2. one forward pass succeeds;
3. flow-matching MSE is finite;
4. backward succeeds;
5. gradients are finite;
6. optimizer step succeeds;
7. scheduler step succeeds;
8. full checkpoint save succeeds;
9. checkpoint integrity validation succeeds;
10. checkpoint reload succeeds;
11. resumed next optimizer step produces sane loss;
12. short generation sanity check succeeds.

Then run:

```text
10 optimizer steps
```

before attempting:

```text
100 optimizer steps
```

The 6000-step production run remains blocked until both are green.


## Training objective

The production training objective is:

**flow-matching MSE plus normalized-coordinate pose-consistency Huber**

The main production/control branch uses `lambda_pose = 0.04`. Preserve the
recorded pose timestep exposure behavior and resume semantics. Historical
anneal branches may intentionally vary `lambda_pose`; they are not a reason to
redefine the main recipe.

Do not add:

- PCK loss;
- perceptual pose loss;
- CLIP loss;
- auxiliary losses other than the canonical pose-consistency Huber;
- image similarity loss;
- additional experimental objectives;

without explicit user approval.

Pose metrics are for evaluation only.


## Timestep sampling

The production flow-matching implementation must explicitly define and test:

- timestep distribution;
- logistic-normal sampling if retained;
- resolution-dependent timestep shift if retained;
- construction of noisy image latent;
- clean control latent behavior;
- flow target.

Do not copy stale toy-test behavior into production without verifying the intended math.


## Optimizer and scheduler

Current baseline unless explicitly changed:

```text
AdamW
lr = 1e-4
betas = (0.9, 0.99)
weight_decay = 0.0
warmup = 200 optimizer steps
```

Target effective batch:

```text
32
```

Do not hard-code a GH200 microbatch before profiling.

The final optimizer choice may only be changed through an explicit experiment/training decision.

Optimizer and scheduler state must be included in production checkpoints.


## Text conditioning

Before production:

- verify the intended tokenizer;
- verify the intended Qwen text/model dependency;
- verify token indices/prefix/suffix assumptions;
- verify selected hidden-layer shapes;
- verify caption dropout;
- verify max-length handling;
- ensure captions are never silently replaced with empty strings because of path/schema mismatch.


## Evaluation

Training optimization uses flow-matching MSE plus the canonical normalized-coordinate pose-consistency Huber.

Evaluation may include:

- validation flow loss;
- sparse generation checks;
- fixed diagnostic prompt/control panel;
- PCK;
- normalized mean error;
- person recall;
- person-count accuracy;
- multi-person matching diagnostics;
- image-quality/style metrics where appropriate.

For pose evaluation:

- recover keypoints from generated RGB;
- compare to control keypoints;
- normalize geometric error by person scale;
- support multi-person matching;
- missing generated people must count against the result.

Evaluation code must not alter the training loss.


## W&B requirements

W&B must be useful but **must never be capable of killing training because the network is temporarily unavailable**.

Log at minimum:

- train loss;
- validation flow loss when run;
- learning rate;
- global gradient norm;
- selected ControlInputLayer gradient norms during diagnostics;
- selected LoRA gradient norms during diagnostics;
- global optimizer step;
- sec/step;
- samples/sec;
- CUDA allocated memory;
- CUDA reserved memory;
- CUDA peak memory;
- control latent RMS/std at diagnostic cadence;
- checkpoint step/time;
- HF upload success/failure;
- age of newest remotely confirmed checkpoint;
- generated diagnostic images at sparse cadence.

Also maintain a lightweight local:

- JSONL;
- or CSV;

metrics log.

W&B must not be the only copy of training telemetry.

Catch and treat as non-fatal:

- login failures;
- initialization failures;
- temporary API failures;
- logging failures;
- transient network failures.

Before production, create a real W&B test run and verify that required metrics/images appear remotely.


## Checkpoint + Hugging Face fail-safe

Recovery objective:

**no more than approximately one hour of training lost under a recoverable outage**

Do not merely upload whatever step checkpoint already exists once per hour.

The training process must **create a fresh full resumable checkpoint on a wall-clock cadence**, default:

```text
60 minutes
```

and then mirror it off-machine.

A full production resume checkpoint must include enough state to continue training:

- trainable model state;
- rank-64 LoRA state;
- `ControlInputLayer` state;
- optimizer state;
- scheduler state;
- global optimizer step;
- gradient-accumulation position if applicable;
- Python RNG state;
- NumPy RNG state;
- torch CPU RNG state;
- torch CUDA RNG state;
- sampler/data-order state or sufficient deterministic state to reconstruct it;
- run configuration;
- code/git commit identifier;
- any state necessary to resume consistently.

Requirements:

- write locally atomically;
- prefer temporary path then validation then rename;
- validate checkpoint integrity;
- keep at least the newest two known-good local checkpoints;
- upload to a **private Hugging Face model repository**;
- upload using retrying non-blocking/background behavior;
- HF failure must not crash training;
- mark a checkpoint remotely confirmed only after successful upload;
- track age of newest remotely confirmed checkpoint;
- on graceful shutdown/error, attempt emergency local checkpoint creation;
- after emergency local save, attempt best-effort HF upload;
- support `--resume auto`;
- `--resume auto` selects newest valid local checkpoint first;
- if local state is unavailable, permit recovery from newest valid HF checkpoint;
- verify integrity before resuming.

A catastrophic host/disk failure may roll training back to the newest successfully mirrored HF checkpoint.

Target maximum rollback:

```text
<= approximately 1 hour
```


## Signal handling

Production training must explicitly handle at least:

- SIGTERM;
- SIGINT.

The handler should trigger controlled shutdown logic:

```text
signal received
    ↓
stop starting new optimizer work
    ↓
attempt emergency local checkpoint
    ↓
best-effort HF mirror
    ↓
flush local metrics/logs
    ↓
exit
```

Signal-handling bugs must be tested before production.


## Unattended execution

Codex is for:

- implementation;
- audit;
- tests;
- debugging.

**Codex is not the production training supervisor.**

Production training should run under a service manager, preferably `systemd`.

The production service should provide:

- restart on process failure;
- short restart delay;
- launch with `--resume auto`;
- logging through journald and/or local files;
- clean working directory;
- correct environment activation;
- correct HF/W&B credential availability;
- graceful SIGTERM handling;
- restart behavior after host reboot when persistent disk survives;
- finite restart guard so a structural deterministic crash does not burn GPU indefinitely.

Before production:

1. run a short training job;
2. kill the process intentionally;
3. confirm emergency checkpoint behavior;
4. confirm systemd restarts the process;
5. confirm `--resume auto` selects the proper checkpoint;
6. confirm training resumes from the expected optimizer step.


## Production launch gate

Do not start the 6000-step run until all are green:

- environment verification;
- uv/dependency environment;
- physical HF dataset resolution;
- immutable manifest validation;
- shard/preprocessing verification;
- real Krea-2 Raw checkpoint load;
- text-conditioning verification;
- VAE verification;
- rank-64 LoRA verification;
- frozen-backbone assertion;
- control-path gradient proof;
- finite production flow loss;
- optimizer/scheduler mechanics;
- 10-step training run;
- 100-step stable training run;
- validation flow loss;
- sparse generation sanity check;
- W&B remote test;
- local metrics fallback;
- full checkpoint save;
- atomic checkpoint validation;
- checkpoint reload;
- HF checkpoint upload;
- HF checkpoint download/reload;
- SIGTERM emergency checkpoint;
- deliberate crash/resume;
- systemd restart/resume;
- disk-space verification;
- final production configuration printed;
- final production configuration committed to Git.

When all gates are green:

**stop.**

Present:

- final training config;
- exact production command;
- expected checkpoint paths;
- W&B run/project;
- HF backup target;
- service command/status instructions.

Wait for explicit user approval before starting the paid 6000-step production run.


## Git workflow

Codex should normally modify and test files but leave repository-history decisions to the user.

Default:

```text
Codex edits/tests
    ↓
updates CODEX_HANDOFF.md
    ↓
stops
    ↓
user reviews in lazygit
    ↓
user stages
    ↓
user commits
    ↓
user pushes
```

Do not automatically commit or push unless explicitly authorized for the session.

Before session completion after code/config changes:

```bash
git diff --check
git status --short
```

Do not place secrets, datasets, model weights, checkpoints, generated caches, or credentials into Git.


## Definition of done

This project is complete when the rank-64 skeleton-conditioned Krea-2 Raw Pose Control-LoRA can:

- train from the production configuration;
- demonstrably receive useful gradients through the pose-control channel;
- keep the base backbone frozen as intended;
- optimize only the intended trainable modules;
- run the intended flow-matching MSE objective;
- consume the immutable dataset splits correctly;
- maintain synchronized RGB/control preprocessing;
- log robustly to W&B;
- preserve local telemetry if W&B is unavailable;
- create atomic fully resumable checkpoints;
- mirror checkpoints to private HF storage on approximately hourly cadence;
- recover automatically from a forced process death;
- resume optimizer/scheduler/data/RNG state correctly;
- survive short stability runs without NaNs;
- survive short stability runs without OOMs;
- avoid corrupted checkpoints;
- avoid silent data failures;
- avoid silent control-path failures;
- pass the 10-step and 100-step gates;
- pass the final production-readiness review;
- complete the approved production run.

There are no Phase 2 or Phase 3 deliverables.

Do not spend implementation effort, repository complexity, or Codex context on cancelled RGB-reference adapter work.
