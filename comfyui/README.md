# Krea-2 Pose Control-LoRA v1 for ComfyUI

This standalone ComfyUI package serves frozen `krea2-pose-control-mix025.safetensors`.
Copy `krea2_pose_control/` directly into `ComfyUI/custom_nodes/krea2_pose_control/`.
It vendors the needed Krea-2 model surgery and Turbo sampler: no import reaches
`/home/ubuntu/krea2-pose-controlnet`, no NFS checkpoint is used, and this source
repository is not needed after copying.

## Installation

Use ComfyUI's Python environment and retain its existing Torch/CUDA build.

```bash
cd ComfyUI
python -m pip install "safetensors>=0.5" "diffusers>=0.33" "transformers>=4.51" "huggingface_hub>=0.30" "einops>=0.8" "Pillow>=10" "numpy>=1.26"
```

For the default DWPose backend, install **ComfyUI-ControlNet-Aux** through ComfyUI Manager, including its DWPose/ONNX dependencies, and restart ComfyUI. This package imports its structured detector API only when DWPose is selected; it does not import an original DWPose checkout or any repo-local path. For standalone `python -m krea2_pose_control.preflight`, it automatically discovers the sibling `<ComfyUI>/custom_nodes/comfyui_controlnet_aux/src` directory. Do **not** add that `src` directory to `PYTHONPATH` manually. `keypoint_rcnn` remains available as a fallback and requires `torchvision` with `keypointrcnn_resnet50_fpn`.

The DWPose files are downloaded through the normal ControlNet-Aux checkpoint/cache mechanism on first use. Keep them on local persistent storage (not NFS):

```text
<ControlNet-Aux checkpoint root>/yzd-v/DWPose/yolox_l.onnx
<ControlNet-Aux checkpoint root>/yzd-v/DWPose/dw-ll_ucoco_384.onnx
```

The default root is owned by the installed ControlNet-Aux node. Set its documented `AUX_ANNOTATOR_CKPTS_PATH` configuration if the cache must live elsewhere. Qwen-Image VAE and Qwen3-VL-4B-Instruct download/cache on first generation. Authenticate with Hugging Face before using a private artifact.

## Model files

```text
ComfyUI/models/krea2/
  krea2_turbo.safetensors             # required inference base
  krea2_raw.safetensors               # optional provenance-path validation
ComfyUI/models/krea2-pose-control/
  krea2-pose-control-mix025.safetensors
```

Set absolute paths in `Krea2PoseGenerate`. The release may also be `hf://owner/repository/path/krea2-pose-control-mix025.safetensors`. Before loading, it verifies:

```text
SHA-256     6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1
tensors     450
parameters  215,488,512
```

## Workflows

Import either JSON in `comfyui/workflows/`.

- `krea2_pose_from_reference.json`: Load Image → Krea2PoseExtractor → Preview condition → Krea2PoseGenerate → Save Image.
- `krea2_pose_from_condition.json`: Load existing condition → Krea2PoseCondition → Preview condition → Krea2PoseGenerate → Save Image.

`Krea2PoseCondition` validates and returns a valid control IMAGE unchanged. It cannot establish pixel provenance, so give it an existing project control or an output of `Krea2PoseExtractor`.

## COCO-17-only extraction

`Krea2PoseExtractor` defaults to `dwpose`, recommended for stylized, fantasy, anime, illustration, and realistic human references. `keypoint_rcnn` is the fallback backend. Both expose configurable person confidence, keypoint confidence, and `max_people`.

The DWPose path is deliberately:

```text
reference RGB -> DWPose structured whole-body joints -> COCO-17 body normalization
              -> frozen PoseBridge renderer -> Krea-2 Pose Control generation
```

It preserves the source image width, height, and pixel coordinates. Per-person DWPose body confidences rank/filter people; low-confidence or missing physical joints are made absent before rendering. It reads only nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles in the explicit COCO-17 order. It never reads DWPose's synthesized neck, hands/fingers, foot-detail joints, or dense face landmarks.

The frozen renderer then applies the historic COCO-17-to-Body-18 mapping, synthesizes its neck only from retained left/right shoulders, and draws 17 rainbow limbs at 3px followed by white radius-4 endpoints.

**Do not pass a generic OpenPose or native DWPose condition raster to the model.** The release was trained only on the exact frozen PoseBridge representation above. There is no hand/finger control or dense face-landmark control; COCO face points are only nose, eyes, and ears. Missing/low-confidence joints are omitted rather than invented.

## Frozen defaults

- Krea-2 Turbo
- 8 steps
- CFG 0 (fixed internally)
- `mu=1.15`, resolution-dependent mu disabled
- control scale `1.0`
- native/aspect-derived geometry
- no Style-LoRA

`explicit_size` is available when width/height are multiples of 16. Native mode preserves the pose canvas before VAE alignment.

## Preflight without generation

From the copied node directory or this repository's `comfyui/` directory:

```bash
python -m krea2_pose_control.preflight \
  --reference-image /path/to/reference.jpg \
  --release-artifact /path/to/krea2-pose-control-mix025.safetensors \
  --turbo-path /path/to/krea2_turbo.safetensors \
  --raw-path /path/to/krea2_raw.safetensors \
  --backend dwpose \
  --device cuda \
  --dwpose-model-dir /local/models/controlnet_aux \
  --output-condition /tmp/krea2_pose_condition.png
```

`--dwpose-model-dir` is optional; when supplied it is the ControlNet-Aux checkpoint root containing `yzd-v/DWPose/`. The command loads the selected detector, writes only the frozen condition PNG, and prints the retained person count and COCO-17 body-joint count per person. It does not load or run generation. Use `--backend keypoint_rcnn` to exercise the fallback.

For a copied node, `python -m krea2_pose_control.preflight` finds the sibling ControlNet-Aux source automatically. For a source-tree invocation, include only ComfyUI's `custom_nodes` root and this repository's `comfyui` root in `PYTHONPATH`; never add `comfyui_controlnet_aux/src` yourself.

## GH200 extraction-smoke commands (no generation)

Place the four original reference RGB files at the paths below before running. The input files were not retained in this repository, so these commands intentionally do not invent replacements. They write only condition PNGs and JSON console reports under `docs/comfyui/dwpose-smoke/`.

```bash
cd /path/to/ComfyUI/custom_nodes
export KREA2_POSE_REPO=/path/to/krea2-pose-controlnet
export DWPOSE_CACHE=/local/models/controlnet_aux
mkdir -p "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke"

python -m krea2_pose_control.preflight --backend dwpose --device cuda --dwpose-model-dir "$DWPOSE_CACHE" --release-artifact /local/models/krea2-pose-control/krea2-pose-control-mix025.safetensors --turbo-path /local/models/krea2/krea2_turbo.safetensors --raw-path /local/models/krea2/krea2_raw.safetensors --reference-image "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/elegant-wandering-knight-reference.png" --output-condition "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png"
python -m krea2_pose_control.preflight --backend dwpose --device cuda --dwpose-model-dir "$DWPOSE_CACHE" --release-artifact /local/models/krea2-pose-control/krea2-pose-control-mix025.safetensors --turbo-path /local/models/krea2/krea2_turbo.safetensors --raw-path /local/models/krea2/krea2_raw.safetensors --reference-image "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/psychedelic-swordswoman-reference.png" --output-condition "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/psychedelic-swordswoman-condition.png"
python -m krea2_pose_control.preflight --backend dwpose --device cuda --dwpose-model-dir "$DWPOSE_CACHE" --release-artifact /local/models/krea2-pose-control/krea2-pose-control-mix025.safetensors --turbo-path /local/models/krea2/krea2_turbo.safetensors --raw-path /local/models/krea2/krea2_raw.safetensors --reference-image "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/comic-fashion-reference.png" --output-condition "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/comic-fashion-condition.png"
python -m krea2_pose_control.preflight --backend dwpose --device cuda --dwpose-model-dir "$DWPOSE_CACHE" --release-artifact /local/models/krea2-pose-control/krea2-pose-control-mix025.safetensors --turbo-path /local/models/krea2/krea2_turbo.safetensors --raw-path /local/models/krea2/krea2_raw.safetensors --reference-image "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/dark-fantasy-jester-reference.png" --output-condition "$KREA2_POSE_REPO/docs/comfyui/dwpose-smoke/dark-fantasy-jester-condition.png"
```

Package-specific memory use is not yet verified; the known production host is a 96 GB GH200 and needs headroom for Turbo, Qwen3-VL, Qwen VAE, and the selected detector.

## Exact GH200 DWPose extraction verification

This invokes the source-tree package from its `comfyui/` working directory, so it uses the newly shipped discovery code while retaining only the two required `PYTHONPATH` entries. It does not generate an image.

```bash
cd /home/ubuntu/krea2-pose-controlnet/comfyui
PYTHONPATH=/home/ubuntu/ComfyUI/custom_nodes:/home/ubuntu/krea2-pose-controlnet/comfyui \
/home/ubuntu/ComfyUI/.venv/bin/python -m krea2_pose_control.preflight \
  --backend dwpose --device cuda \
  --dwpose-model-dir /home/ubuntu/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts \
  --release-artifact /lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors \
  --turbo-path /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --raw-path /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --reference-image /home/ubuntu/krea2-pose-controlnet/docs/comfyui/dwpose-smoke/elegant-wandering-knight-reference.png \
  --output-condition /home/ubuntu/krea2-pose-controlnet/docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png
```

## One headless frozen-generation smoke

`krea2_pose_control.generation_smoke` is a one-shot wrapper over the exact `Krea2PoseGenerate` runtime. It keeps the frozen release contract: native/aspect geometry from the 832×1216 condition, seed 42, 8 Turbo steps, CFG 0, `mu=1.15`, control scale 1.0, and no Style-LoRA. It does not alter model files or release artifacts.

```bash
cd /home/ubuntu/krea2-pose-controlnet/comfyui
PYTHONPATH=/home/ubuntu/ComfyUI/custom_nodes:/home/ubuntu/krea2-pose-controlnet/comfyui \
/home/ubuntu/ComfyUI/.venv/bin/python -m krea2_pose_control.generation_smoke \
  --pose-condition /home/ubuntu/krea2-pose-controlnet/docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png \
  --release-artifact /lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors \
  --turbo-path /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --raw-path /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --output-image /home/ubuntu/krea2-pose-controlnet/docs/comfyui/generation-smoke/elegant-wandering-knight-seed42.png
```

The default prompt is the frozen elegant-wandering-knight prompt used for this smoke. Pass `--prompt` only to test another prompt; the sampler settings remain pinned.
