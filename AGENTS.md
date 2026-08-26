# Krea-2 Pose Control-LoRA — Phase 1 Agent Instructions

## Mission
Get **Phase 1 only** ready for a production training run on one Lambda GH200:

**skeleton control image -> clean control latent -> channel-concat with noisy image latent -> expanded input projection -> Krea-2 Raw + rank-64 LoRA -> flow-matching target**.

Do not work on RGB-reference adapters or later research phases.

The reference depth-ControlNet repository is a **third-party implementation by Tanmay**, not Krea's official training recipe. Use it as a behavioral reference only. Our training/data/logging/checkpoint code should be project-owned.

## Non-negotiable decisions
- Base model for training: **Krea-2 Raw**.
- Control: rendered pose skeleton, not RGB reference.
- Control mechanism: spatially aligned **channel concatenation**, not extra tokens and not a classical side-branch ControlNet.
- Image latent is noisy at sampled flow timestep; control latent remains clean.
- LoRA rank: **64**.
- Training loss: **flow-matching MSE only**. PCK is evaluation-only.
- Seed: **42**.
- Baseline optimizer unless explicitly changed by the user: AdamW, `lr=1e-4`, betas `(0.9, 0.99)`, weight decay `0.0`.
- Warmup: 200 optimizer steps.
- Target effective batch: 32; choose GH200 microbatch only after measuring VRAM/throughput.
- Baseline max steps: 6000.
- Caption dropout: 0.10.
- Use **uv**, not pip, for project dependency management.
- Do not replace Lambda's working CUDA/PyTorch/cuDNN/Triton stack until it has been inspected and there is a demonstrated incompatibility.

## Data invariants
Treat source data and immutable manifests as read-only.

Expected clean split after the known exclusions:
- train: 16,503
- representative val: 889
- diagnostic val: 24
- total used: 17,416

Raw source/control pairs share stems but use different extensions. Manifest records use the real schema:
- immutable manifest `file_name`: bare `<stem>.jpg`
- physical RGB/control locations on Hugging Face are storage/shard paths and must be discovered after download
- controls are matched by stem and use `.png`
- `text`: caption

Do not infer physical paths from manifest filenames. Build a read-only stem → physical RGB/control index from the downloaded HF snapshot and validate it against the immutable manifests.

Normalize path/basename joins deliberately. Never silently replace a missing caption with `""`.

RGB and skeleton must undergo **the exact same resize/crop/bucket geometry**.

## Important reference behavior
The reference `ControlInputLayer` doubles the input feature width while preserving token count. Its initialization copies pretrained image weights into the first half and initializes the control half to zero. Therefore:

**At initialization, changing the skeleton should NOT change the model output. This is expected.**

What must be verified instead:
1. control tensors are non-empty, finite, correctly aligned, and have the expected shape;
2. after the first backward pass, the gradient norm of the control half of `ControlInputLayer.weight` is finite and **> 0**;
3. after one or more optimizer steps, real-control vs zero-control forward passes with otherwise identical inputs begin to diverge;
4. LoRA trainable tensors receive finite gradients;
5. frozen backbone tensors remain frozen.

Do not incorrectly declare the control path broken merely because step-0 outputs are identical across controls.

## Repo scan — do this before edits
Start read-only. Do not recursively dump the entire repository into context.

1. `git status --short` and `git log -5 --oneline`.
2. `rg --files` while excluding `.git/`, `.venv/`, `data/`, `checkpoints/`, `outputs/`, caches, model weights, shards, images, and generated artifacts.
3. Locate and inspect only relevant files/symbols first:
   - `train.py`
   - `prepare_shards.py`
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
   - `scripts/prefetch_models.py`
   - smoke-test scripts
   - `base_model/mmdit.py` and `base_model/k2_lora.py`
   - reference `k2_lora.py`, `mmdit.py`, `trainer/sampling.py`, and `trainer/train_control_lora.py` only when needed to verify architecture/math.
4. Produce a short audit: `PASS / FIX / BLOCKED` for environment, data loader, control path, loss math, optimizer/scheduler, W&B, checkpoint/resume, HF backup, signal handling, and unattended service.
5. Only then edit code. Prefer small reviewable changes and test each layer before moving on.


## Hands-off session operating protocol
Default behavior for every Codex session is **hands-off execution within the stated task boundary**. Do not stop after each minor step to ask for permission. Inspect, implement, run targeted tests, diagnose failures, repair them, and re-test until the bounded task is complete or a genuine blocker is reached.

### At session start
1. Read this `AGENTS.md`.
2. Read `docs/CODEX_HANDOFF.md` if it exists. Treat it as the current operational state, not as an unquestioned source of truth; verify material claims against code/tests when relevant.
3. Run `git status --short` before edits. Never overwrite unrelated user changes.
4. Identify the single bounded session objective from the user prompt. Do not expand scope unless required to make that objective work.
5. Use targeted search (`rg`, `rg --files`) and open only the files needed for the objective.

### During the session
- Work autonomously through normal implementation details, package installation allowed by the task, test failures, small refactors, and documentation updates.
- Prefer the smallest coherent patch that satisfies the acceptance criteria.
- After each material change, run the narrowest useful test before moving on.
- If a test fails, investigate and fix it rather than immediately returning the failure to the user.
- Escalate only for a **genuine blocker**: missing credential/access, missing external artifact, destructive/irreversible action not already authorized, contradictory requirements, or a decision that materially changes the training experiment.
- Never treat transient W&B/HF/network failures as reasons to stop training-system implementation; add retry/nonfatal behavior where required.
- Never change the working torch/CUDA/cuDNN/Triton/NVIDIA stack without demonstrated incompatibility and explicit approval.
- Never launch the paid 6000-step production run without explicit user approval. This is the primary exception to hands-off execution.

### Model/effort routing
Keep expensive reasoning focused on tasks that need it. When the installed Codex environment supports model selection/delegation:
- **Luna / low reasoning:** installations, version checks, file discovery, simple config edits, formatting, mechanical validations.
- **Terra / medium reasoning:** bounded Python implementation, dataset indexing, preprocessing, logging, checkpoint plumbing, ordinary debugging and tests.
- **Sol / high reasoning:** architecture/math changes, control-path correctness, difficult numerical/CUDA issues, optimizer/training decisions, final production-readiness review.

Do not use Sol merely to run shell commands or perform mechanical edits. Do not spawn multiple agents for work that one bounded task can complete.

### Session completion — mandatory handoff
Before ending **every session that inspects or changes project state**, update `docs/CODEX_HANDOFF.md`. Create it if absent. This update is part of the task and should happen automatically without being requested.

`docs/CODEX_HANDOFF.md` must stay concise (target roughly 2–5 KB) and contain only current state:
- current Phase-1 objective;
- verified environment facts;
- decisions currently in force;
- completed/green gates;
- current failures/blockers;
- files changed this session;
- exact tests/commands run with PASS/FAIL;
- important new findings;
- exact next recommended action.

Rewrite stale sections instead of appending a chronological transcript. Do not paste large logs, diffs, stack traces, source files, or chat history. A fresh Codex session should be able to continue by reading only `AGENTS.md`, `docs/CODEX_HANDOFF.md`, and files relevant to its new task.

If code/config was changed, finish with:
1. targeted tests green or clearly documented blocker;
2. `git diff --check`;
3. `git status --short`;
4. concise handoff update.

Do not automatically commit or push unless the session/user instruction explicitly authorizes Git writes. Never include credentials, tokens, model weights, datasets, checkpoints, or generated secrets in Git.

## Phase 1 verification gates
Do not start the 6000-step run until every gate is green.

### Gate A — GH200 environment
- Linux ARM64 detected.
- CUDA visible.
- BF16 supported.
- PyTorch/cuDNN/Triton versions recorded.
- SDPA/attention smoke test passes.
- `uv` environment works without accidentally replacing the known-working system CUDA stack.

### Gate B — data and preprocessing
- Manifest schema matches actual files.
- Train/val/diagnostic counts match expected counts.
- All source/control pairs exist and align by stem.
- Shards open successfully and contain finite image/control latents and captions.
- Control latent has measurable signal (RMS/std/nonzero statistics logged for smoke runs).
- RGB/control shapes are identical after preprocessing.

### Gate C — control-path proof
Add a short diagnostic mode/test that logs, without changing the production objective:
- image latent RMS/std;
- control latent RMS/std;
- concatenated tensor shape;
- `ControlInputLayer` image-half weight norm;
- control-half weight norm;
- control-half gradient norm after backward;
- representative LoRA A/B gradient norms.

Run the same batch with a real control and a zero control:
- before training: output equality is expected because control-half weights are zero-initialized;
- after optimizer update(s): output difference should become non-zero and finite.

Fail loudly on NaN/Inf, shape mismatch, empty controls, zero control-half gradients after backward, or accidental trainability of the frozen backbone.

### Gate D — training mechanics
Prove in order:
1. one real batch loads;
2. one forward pass;
3. flow-matching MSE is finite;
4. backward pass;
5. optimizer step;
6. scheduler step;
7. checkpoint save;
8. checkpoint reload;
9. resumed next optimizer step gives sane loss;
10. short generation sanity check.

Then run 10 steps and 100 steps before production.

## W&B requirements
W&B must be useful but **must never be able to kill training because the network is temporarily unavailable**.

Log at minimum:
- train loss;
- validation flow loss when run;
- learning rate;
- global grad norm;
- selected ControlInputLayer/LoRA grad norms during diagnostics;
- step;
- sec/step and samples/sec;
- CUDA allocated/reserved/peak memory;
- control latent RMS/std (diagnostic cadence, not every step);
- checkpoint step/time;
- HF upload success/failure and age of newest remotely confirmed checkpoint;
- generated diagnostic images at the chosen sparse cadence.

Also write a lightweight local JSONL/CSV metrics log so W&B connectivity is not the only copy of basic training telemetry. W&B exceptions/retries must be caught and treated as non-fatal.

Before production, create a real W&B test run and confirm metrics/images appear remotely.

## Checkpoint + Hugging Face fail-safe
The recovery target is **<= 1 hour of lost training** under a recoverable outage.

Do not merely upload the latest step-based checkpoint every hour. The code must **create a new full resume checkpoint on a wall-clock cadence (default 60 minutes)** and then upload it.

A production resume checkpoint must include enough state to continue training, not just inference weights:
- trainable model state (LoRA + ControlInputLayer);
- optimizer state;
- scheduler state;
- global optimizer step;
- gradient-accumulation position if relevant;
- RNG states (Python, NumPy, torch CPU/CUDA);
- sampler/data-order state or enough deterministic state to reconstruct it;
- run config and code/git commit identifier.

Requirements:
- save locally **atomically** (`tmp` then rename);
- keep at least the newest 2 known-good local checkpoints;
- upload to a **private HF model repo** in a retrying background worker/process;
- never block or crash training because HF is temporarily unreachable;
- mark a checkpoint remotely confirmed only after upload succeeds;
- on shutdown/error, attempt an emergency local checkpoint and best-effort HF upload;
- on launch, `--resume auto` must select the newest valid local checkpoint, or newest downloaded HF checkpoint when local state is unavailable;
- verify checkpoint integrity before resuming.

A Lambda/host outage can still stop computation. `systemd` can restart the process only if the machine returns. HF is the off-machine recovery copy, so a catastrophic host/disk loss may roll back to the newest successfully uploaded checkpoint (target <= 1 hour).

## Unattended execution
Codex is for implementation, audit, and tests. **Codex is not the training supervisor.**

Production training should be run by a service manager, preferably `systemd`:
- restart on failure;
- enabled on boot when appropriate;
- short restart delay;
- logs to journald and/or a local file;
- graceful SIGTERM/SIGINT handler triggers emergency checkpoint;
- `--resume auto` on every launch;
- finite restart guard so a structural crash does not burn GPU indefinitely.

Before production, deliberately kill the 100-step test mid-run and prove service restart + automatic resume works.

## Production launch gate
Do not start the 6000-step run until all are true:
- environment tests green;
- data/shard verification green;
- control gradient proof green;
- 10-step run green;
- 100-step run stable;
- W&B remote test green;
- HF full-resume checkpoint upload/download/reload green;
- deliberate SIGTERM/crash resume test green;
- service restart test green;
- disk-space check green;
- final config printed and committed to Git.

When ready, stop and present the exact production command/config for explicit user approval before launching the paid 6000-step run.

## Codex efficiency / token discipline
- Keep this `AGENTS.md` concise and stable; do not paste historical chat transcripts into it.
- Use `rg`/targeted symbol search before opening files.
- Read slices around relevant definitions instead of whole large files.
- Never scan image data, model checkpoints, latent shards, `.venv`, or generated outputs unless a specific test requires one sample/file.
- Use existing handoff docs as references, but search/open only the section needed for the current question.
- Maintain `docs/CODEX_HANDOFF.md` as the single short cross-session state file. Do not create competing status/handoff documents unless explicitly requested.
- Prefer one bounded Codex task per session: audit, preprocessing fix, control test, logging, checkpoint/resume, or service setup. New session after a major milestone keeps context smaller.
- Ask Codex for a plan/audit first for risky multi-file changes, then implement the smallest coherent patch.
- Avoid automatic review on every tiny change; it consumes additional model calls. Use it at milestone/release boundaries.
- Keep command output summarized; redirect verbose logs to files and inspect errors/tails.
- Do not repeatedly re-read unchanged files. Refer to git diff and the status doc.
- Do not use `--dangerously-bypass-approvals-and-sandbox`. For implementation, prefer workspace-write with on-request approvals; production training itself should run outside Codex under the supervisor.

## Definition of done for Phase 1
Phase 1 is done when the rank-64 skeleton-conditioned Krea-2 Raw Control-LoRA can start from the production config, demonstrably receive/learn from the pose control channel, log robustly to W&B, create and mirror fully resumable hourly checkpoints to HF, survive a forced process death via automatic resume, and complete short stability runs without NaNs, OOMs, corrupted checkpoints, or silent data/control failures.
