# Frozen training recipe

Source: `docs/blog-evidence/TRAINING_METHODS_INFRA.md; docs/blog-evidence/training_methods_infra.json`.

| Setting | Frozen value |
| --- | --- |
| Base model | Krea-2 Raw |
| LoRA rank | 64 |
| Trainable parameters | 215,488,512 float32 |
| Trainable tensor count | 450 |
| ControlInputLayer parameters | 792,576 |
| LoRA parameters | 214,695,936 |
| Optimizer | AdamW |
| Betas | (0.9, 0.99) |
| Epsilon | 1e-8 |
| Weight decay | 0.0 |
| Max grad norm | 1.0 |
| Microbatch | 1 |
| Gradient accumulation | 32 |
| Effective batch | 32 |
| Caption dropout | 0.10 |
| Control dropout | 0.0 |
| Precision | CUDA BF16 autocast |
| Workers / prefetch | 4 / 4 (persistent workers; pin memory) |
| Bucket scheme | 768×768, 704×896, 896×704, 640×960, 960×640, 576×1024, 1024×576, 512×1152, 1152×512 |
| Gradient checkpointing | Disabled (0 blocks) |
| torch.compile | Disabled |
| Fused AdamW | Disabled |
