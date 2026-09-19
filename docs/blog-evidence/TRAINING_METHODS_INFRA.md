# Krea-2 Pose Control-LoRA v1 — training methods and infrastructure evidence

Status: frozen evidence record, audited 2026-09-19 UTC. It describes the lineage ending in `mix-025`; it does not reinterpret evaluation outcomes.
`Verified` means direct durable-artifact or host evidence, `Reconstructed`
means a calculation from retained records, and `Unresolved` is not a blog fact.

## Final lineage

`mix-025` is not separately trained. It is float32 interpolation of trainable
`state['model']` tensors only: `0.75 * parent-4000 + 0.25 * finish-control-a4300`.
Audit-time SHA-256 recomputation matched the frozen endpoint values below. [E1–E2] **Verified.**

| Stage | Steps | LR policy | Change/purpose | Selected checkpoint | Runtime evidence |
| --- | ---: | --- | --- | --- | --- |
| `pose-control-production-3000` | 0–3000 | 200-step warmup to `1e-4`, then `1e-4` | initial production | cooldown parent | 12:23:40 metric time; 12:24:47 observed span [E3–E4] |
| `pose-control-production-cooldown-3000-to5000` | 3000–4000 used | cosine `1e-4` → `1e-5` over 2000 updates | continuation; selected parent | `parent-4000` | first 1000: 4:10:16 metric time; 4:10:42 span [E5–E6] |
| `pose-control-finish-control-4000-to4500` | 4000–4300 used | cosine `2e-5` → `5e-6` over 500 updates | constant pose loss weight | `finish-control-a4300` | first 300: 1:17:09 metric time; 1:17:25 span [E7–E8] |
| `mix-025` | n/a | n/a | deterministic endpoint blend | final release | no training runtime [E1] |

The calendar span from the retained initial W&B start to A4300 is 25:06:48,
but includes two inter-run gaps and is not a core-training-time claim. [E4,E6,E8]
**Reconstructed.**

## Infrastructure

| Fact | Frozen value | Confidence / source |
| --- | --- | --- |
| GPU | PyTorch normal-host device name: `NVIDIA GH200 480GB`. Here `480GB` is part of the device/product name, not a GPU-HBM capacity claim. | **Directly verified** [E24] |
| GPU memory | PyTorch-visible total device memory: `101468602368` bytes (**94.5 GiB**). | **Directly verified** [E24] |
| Architecture | ARM64 / `aarch64`. | **Verified** [E10] |
| OS/kernel | Ubuntu 22.04.5 LTS; `6.8.0-1046-nvidia-64k`. | **Verified** [E10] |
| Python | CPython 3.10.12. | **Verified** [E10] |
| PyTorch/CUDA | PyTorch 2.7.0; PyTorch CUDA runtime 12.8; `nvcc` 12.8.93. | **Verified** [E10] |
| cuDNN/Triton/driver | cuDNN 9.8.0 (`90800`); Triton 3.3.0; driver 580.105.08. | **Verified** [E10] |
| uv and venv | audit-time uv 0.12.17; `.venv` inherits system site packages. Torch-family wheels are intentionally absent from the uv project environment. | **Verified** [E10–E11] |
| Storage | repo/venv on local ext4; data, caches, checkpoints, logs, release inputs on read-only NFS `/lambda/nfs/adhit`. | **Verified** [E12] |
| Device setup | single CUDA device / world size 1; no distributed launcher. | **Verified** [E13] |
| Precision | CUDA BF16 autocast; control/text model inputs BF16; clean/noise and MSE comparison float32. | **Verified** [E14] |
| Runtime choices | no compile, no fused AdamW, and gradient checkpointing disabled (`gradient_checkpointing_blocks=0`) for all three final-lineage runs; 4 loader workers, pin memory, persistent workers, prefetch 4; cached text and VAE latents on NFS. Generic `train.py` supports 0–28 checkpointed blocks, but that capability was not enabled for this production lineage. | **Verified** [E3,E5,E7,E13,E17] |

The audit sandbox reports `torch.cuda.is_available() == false` and `nvidia-smi`
fails to communicate with the driver. This is a current sandbox limitation, not
contrary evidence about the normal-host GH200 training record. [E9–E10]

## Frozen training recipe

Base checkpoint: Krea-2 Raw at
`/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`, SHA-256
`f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7`.
[E3] **Verified.**

The final endpoint has **215,488,512 float32 trainable parameters in 450
tensors**: `first.weight`/`first.bias` for the expanded ControlInputLayer
(792,576 parameters) and 214,695,936 LoRA parameters. Rank-64 LoRA A/B
tensors target `attn.{wq,wk,wv,gate,wo}` and `mlp.{gate,up,down}` in each of
28 blocks; the backbone is frozen. [E16] **Verified.**

- AdamW: betas `(0.9, 0.99)`, epsilon `1e-8`, weight decay `0.0`, max-grad
  norm `1.0`. [E3,E17]
- Microbatch `1` × accumulation `32` = effective batch `32`; 4 loader workers.
  [E3]
- Caption dropout `0.10`; control dropout `0.0`; global and flow-generator
  seed `42`. Bucket shuffle and cached unconditional-caption selection are
  deterministically seeded. [E3,E14,E17]
- Local checkpoint cadence: initial/cooldown every 250 steps; finish-control
  every 100; each also had a 3600-second fallback (hence the initial extra
  step-2750 save). HF mirror cadence was 500 steps for initial production,
  250 for cooldown, and 100 for finish-control, to private repo
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`. [E3,E5,E7,E18]
- Native cache: 16,503 examples, VAE factor 8, seeded-per-stem Qwen posterior
  sampling, 256 samples/shard. Buckets: `768x768`, `704x896`, `896x704`,
  `640x960`, `960x640`, `576x1024`, `1024x576`, `512x1152`, `1152x512`.
  [E15]

## Exact methodology and objective

Image and rendered-skeleton control latents are spatially aligned and
patchified independently, then channel-concatenated into the expanded input
projection. Image latent is noised; control latent remains clean. [E14]

For clean latent `x_0`, Gaussian `eps`, and timestep `t`, code uses
`x_t = t*eps + (1-t)*x_0` and velocity target `v = eps - x_0`; flow loss is
float32 MSE over predicted and target patch tokens. [E14,E19] **Verified.**

The normal pre-shift timestep is `sigmoid(N(0,1))`, then
`exp(mu)t / (exp(mu)t + 1 - t)`, where `mu` is linear between sequence-length
anchors `(256, 0.5)` and `(6400, 1.15)`. Auxiliary pre-shift probability and
forced pose-exposure probability are both zero in this lineage. [E3,E14]

Pose loss runs only for pose-reward-available samples whose final shifted time
is in `[0.10, 0.20]`. It forms `x0_hat = image_tokens - t*velocity`,
differentiably VAE-decodes active samples to unit RGB, and uses frozen
`torchvision/keypointrcnn_resnet50_fpn:COCO_V1` with authoritative boxes.
Detector/RPN/box scoring/NMS/argmax never enter the loss path. [E19,E20]

For positive-area authoritative boxes, targets are 17-joint
`joint_provenance[*].training_coordinate`; validity is the matching
`reward_joint_valid` mask. Prediction and target both normalize as
`(xy - (x0,y0)) / max((width,height), eps)`. PyTorch Huber uses `delta=1.0`,
means x/y per valid joint, then means valid joints. Total loss is
`flow_mse + 0.04 * pose_huber` only when active, otherwise flow MSE. [E19–E21]
**Verified.**

The selected parent and finish-control branch retain `lambda_pose=0.04`
constant. The sibling pose-anneal run is not an endpoint and is not part of
`mix-025`. [E1,E7]

## Runtime, cache, and evaluation bounds

Retained JSONL `sec_per_step` totals are 12:23:40 (initial), 8:17:20
(full cooldown), and 2:06:53 (full finish-control). The endpoint-relevant
0–3000 + 3001–4000 + 4001–4300 sum is **17:51:05**. The timer ends before
checkpoint serialization/mirroring; see [RUNTIME_AUDIT.md](RUNTIME_AUDIT.md).
[E4,E6,E8] **Reconstructed.**

768 latent-cache artifact timestamps span 2026-08-30 22:14:38–22:54:59
(40:21); persistent text-cache shard timestamps span 2026-08-26
12:55:27–13:09:04 (13:37). These are artifact-write intervals, not complete
job runtimes. [E15,E22] **Reconstructed.**

The final-val artifact sequence spans at least 2026-09-03 07:09:08–08:44:56
(1:35:48), covering endpoint and three interpolation candidates. Exact
evaluation total and mix-025-only total are unresolved. [E23]

## Evidence registry

| ID | Source / locator |
| --- | --- |
| E1 | `docs/evaluation/release/final_release_v1.json`, `candidate.interpolation`; `FINAL_RELEASE_DECISION.md` |
| E2 | audit-time `sha256sum` of the two NFS endpoint checkpoint files |
| E3 | NFS `checkpoints/pose-control-production-3000/run_metadata.json` |
| E4 | matching initial `metrics.jsonl`, log, and checkpoint mtimes |
| E5 | NFS cooldown `run_metadata.json` |
| E6 | matching cooldown `metrics.jsonl`, log, and checkpoint mtimes |
| E7 | NFS finish-control `run_metadata.json` |
| E8 | matching finish-control `metrics.jsonl`, log, and checkpoint mtimes |
| E9 | `AGENTS.md`, “Verified GH200 environment” |
| E10 | direct 2026-09-19 host audit: `uname`, OS release, NVIDIA proc file, Python, `nvcc`, `nvidia-smi` |
| E11 | `.venv/pyvenv.cfg`, `scripts/create_uv_env.sh`, `pyproject.toml` |
| E12 | direct `findmnt -T` / `df -hT` audit |
| E13 | `pose_controlnet/production_training.py` at checkpoint code revision `c5771ef`: `ProductionRecipe` defaults/locks `gradient_checkpointing_blocks=0`; `build_train_config` passes `gradient_checkpointing=False`, `gradient_checkpointing_blocks=0` |
| E14 | `pose_controlnet/diffusion.py` at `c5771ef` |
| E15 | NFS `posebridge_latents_768/shards.json` and cache mtimes |
| E16 | direct mmap accounting of A4300 `state['model']`; throughput JSON |
| E17 | `train.py` at `c5771ef`: optimizer, bucket, and dropout functions; generic checkpointing range validation `[0,28]` |
| E18 | NFS stage checkpoint listings and `observability.hf` metadata |
| E19 | `scripts/train_pose_reward_smoke.py::_pose_smoke_loss` at `c5771ef` |
| E20 | `pose_controlnet/keypoint_critic.py::FixedBoxKeypointRCNNCritic` at `c5771ef` |
| E21 | `scripts/audit_keypoint_critic.py::_person_tensors`; normalized Huber function at `c5771ef` |
| E22 | NFS text/cache artifact mtimes |
| E23 | NFS final-val artifact mtimes |
| E24 | direct normal-host PyTorch query: device name and `torch.cuda.get_device_properties(0).total_memory` |

## Explicitly unresolved

- Training-time uv executable version (audit-time value is not substituted).
- Complete preprocessing, startup, upload, and evaluation wall-clock totals.
