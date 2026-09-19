# Final-lineage training stages

Source: `docs/blog-evidence/TRAINING_METHODS_INFRA.md` and `docs/blog-evidence/RUNTIME_AUDIT.md`. Runtime is measured optimizer-loop time, not complete wall-clock time.

| Stage | Step range | LR schedule | Pose-loss setting | Selected checkpoint | Optimizer-loop runtime | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| Production | 0–3000 | 200-step warmup to 1e-4, then 1e-4 | Production objective; λ_pose 0.04 when active | Feeds cooldown | 12:23:40 | Initial production |
| Cooldown | 3000–5000 run; parent selected at 4000 | cosine 1e-4 → 1e-5 over 2000 updates | Production objective; λ_pose 0.04 when active | parent-4000 | 3001–4000: 4:10:16 | Continuation / parent selection |
| Finish-control branch | 4000–4500 run; A4300 selected | cosine 2e-5 → 5e-6 over 500 updates | λ_pose = 0.04 constant | finish-control-a4300 | 4001–4300: 1:17:09 | Finish-control selection |
| mix-025 release | n/a | n/a | n/a | final release | No training runtime | 0.75 parent-4000 + 0.25 A4300 trainable-tensor interpolation |
| Endpoint-relevant total | 0–3000 + 3001–4000 + 4001–4300 | — | — | mix-025 inputs | **17:51:05** | Selected endpoint lineage only |
