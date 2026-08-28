# Phase 2 differentiable pose-critic audit

This is audit-only code. It is not imported by `train.py`, does not alter the
flow-MSE objective, does not load a continuation checkpoint, and contains no
optimizer operation. Keypoint R-CNN remains the external evaluation metric.

## Candidate provenance and raw representation

The selected official candidate is OpenMMLab MMPose v1.3.2
`rtmpose-m_8xb64-270e_coco-wholebody-256x192`. Official config:
<https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py>.
Official weights: `rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth` from the OpenMMLab URL embedded in `pose_critic.py`.

The config specifies input `(W,H)=(192,256)`, SimCC split ratio `2.0`, x/y
vectors `(384,512)`, Gaussian sigmas `(4.9,5.66)` in split-bin units, and raw
head outputs before decode. The audit consumes only output indices 0–16 (COCO
physical body joints). It excludes synthetic neck, face, hands, and foot-extra
joints. For a compatible host, stage the config and weight explicitly, then
record `sha256sum WEIGHTS.pth` in the output JSON; this sandbox could not
download it, so no SHA is asserted or fabricated.

The non-random fixed crop exactly follows MMPose validation geometry:
`GetBBoxCenterScale(padding=1.25)` then `TopdownAffine(input=192x256,
use_udp=False)`: expand the sidecar final-frame `bbox_training_xywh`, enforce
aspect ratio 192/256, and bilinearly grid-sample with zero OOB padding. The
same affine maps sidecar COCO-17 coordinates into crop coordinates. A joint is
eligible only when sidecar reward provenance says so and its SimCC coordinate
lies in `[0, vector_length)`; source/final-frame OOB joints are excluded.

The audit candidates are (1) softmax-expectation Huber coordinate loss and
(2) cross entropy against the config-faithful Gaussian SimCC target. Neither
uses argmax.

## Required host dependency check

No packages were installed in this audit shell: PyPI DNS resolution failed.
On the verified GH200 shell, first use this exact non-Torch stack and confirm
that it leaves Torch 2.7.0/CUDA 12.8 untouched:

```bash
uv add 'mmengine==0.10.7' 'mmcv-lite==2.1.0' 'mmdet==3.3.0' 'mmpose==1.3.2'
```

Then stage official files explicitly and run:

```bash
uv run python scripts/audit_pose_critic.py --sidecar /tmp/pose_targets_v3_authoritative_20260828 --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --config /path/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py --weights /path/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth --output-dir /tmp/pose-critic-real
```

The script deterministically takes the first 16 eligible sidecar rows in each
of COCO, Human-Art painting, real-human, and sculpture; reports normalized
coordinate error, PCK-like .05/.10 rates, raw confidence/entropy, both losses,
and one image-gradient check per source. Outputs remain under `/tmp`.

## Current blocker

This Codex sandbox has no visible CUDA device and cannot resolve PyPI,
GitHub, or OpenMMLab. Therefore RTMPose weights/config, real-image agreement,
VAE decode-autograd, x0-hat timestep, and GH200 memory/latency gates have not
been executed. Do not infer feasibility or a training reward region from the
CPU toy tests. The VAE/timestep audit is intentionally blocked until the
real-image and VAE gradient gates run on the GH200 production shell.
