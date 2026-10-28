# Krea-2 Pose Control-LoRA v1 for ComfyUI

This is the completed standalone ComfyUI integration for frozen
`krea2-pose-control-mix025.safetensors`. Copy `krea2_pose_control/` into
`ComfyUI/custom_nodes/krea2_pose_control/`; it vendors the release-scoped
Krea-2 control runtime and does not depend on this source repository or an NFS
training checkpoint after installation.

## Install and model paths

Keep ComfyUI's existing Torch/CUDA installation. Install the package
dependencies in ComfyUI's Python environment:

```bash
cd ComfyUI
python -m pip install "safetensors>=0.5" "diffusers>=0.33" "transformers>=4.51" "huggingface_hub>=0.30" "einops>=0.8" "Pillow>=10" "numpy>=1.26"
```

Install **ComfyUI-ControlNet-Aux** through ComfyUI Manager for the default
DWPose backend, including its documented DWPose/ONNX dependencies. The node
locally discovers ControlNet-Aux's sibling `src` directory when necessary; do
not add that directory to `PYTHONPATH` manually. An optional flash-attn ABI
warning from another ComfyUI component does not block these Krea nodes.

Place the frozen files at these paths (or supply equivalent absolute paths to
the generator node):

```text
ComfyUI/models/krea2/krea2_turbo.safetensors
ComfyUI/models/krea2/krea2_raw.safetensors                  # optional provenance validation
ComfyUI/models/krea2-pose-control/krea2-pose-control-mix025.safetensors
```

The release validates SHA-256
`6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`
before it loads.

## Nodes and workflows

Import either workflow from `comfyui/workflows/`.

- `krea2_pose_from_reference.json`: `Load Image` → `Krea2PoseExtractor` →
  `Krea2PoseGenerate` → `Save Image`.
- `krea2_pose_from_condition.json`: `Load Image` → `Krea2PoseCondition` →
  `Krea2PoseGenerate` → `Save Image`.

`Krea2PoseExtractor` inputs are `reference_image`, `backend`, person/keypoint
confidence thresholds, and `max_people`; `dwpose` is the default and
`keypoint_rcnn` is the fallback. `Krea2PoseCondition` accepts only
`pose_control_image` and validates an existing project-format condition.
`Krea2PoseGenerate` accepts `prompt`, `pose_condition`, `seed`,
`control_scale`, `steps`, `mu`, `geometry_mode`, `width`, `height`,
`release_artifact_path`, `krea2_turbo_path`, and `krea2_raw_path`.

The exact default conversion is:

```text
reference RGB -> DWPose structured Body-18 joints -> COCO-17 normalization
              -> frozen PoseBridge renderer -> Krea2PoseGenerate
```

Only COCO-17 body joints enter PoseBridge. It then applies the frozen historic
COCO-17-to-Body-18 mapping and renderer. Do not feed generic OpenPose or native
DWPose preview rasters directly to the model.

## Frozen generation contract

- Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`
- Control scale 1.0 by default
- No Style-LoRA path
- Frozen `mix-025` release artifact above

`native_aspect` sends the condition canvas dimensions to inference; therefore
the condition width and height must already be positive multiples of 16.
`explicit_size` instead sends the entered `width` and `height`; both must be
positive multiples of 16. These are model-validity requirements, not merely UI
recommendations. Explicit size applies the release resize/center-crop geometry
to the control before encoding.

For a non-generating installation check, run the package's
`krea2_pose_control.preflight` module from the copied node directory. It uses
the same DWPose → COCO-17 → PoseBridge path and validates the selected frozen
model paths without invoking the sampler.
