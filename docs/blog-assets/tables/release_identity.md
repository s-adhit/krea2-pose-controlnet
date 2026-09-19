# Release identity

Primary source: `docs/evaluation/release/final_release_v1.json`. Safetensors public filename and frozen NFS materialization path/SHA are recorded in `docs/release/HF_MODEL_CARD.md` and the frozen canonical-generation summary.

| Field | Frozen value |
| --- | --- |
| Release ID | krea2-pose-control-lora-v1 |
| Candidate | mix-025 |
| parent-4000 path / SHA | `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt` / `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd` |
| A4300 path / SHA | `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt` / `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff` |
| Interpolation formula | 0.75 × parent-4000 + 0.25 × A4300; float32; `state['model']` trainable control/LoRA tensors only |
| Release safetensors filename / public path | `krea2-pose-control-mix025.safetensors` / `release/krea2-pose-control-mix025.safetensors` |
| Release safetensors materialization path | `/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors` |
| Release safetensors SHA256 | `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1` |
| Trainable parameters | 215,488,512 |
| Trainable tensor count | 450 |
| Runtime defaults | Krea-2 Turbo; 8 steps; CFG 0; μ 1.15; control scale 1.0; native cached geometry |
