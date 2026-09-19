# Runtime audit — final Krea-2 Pose Control-LoRA lineage

All timestamps are UTC. This audit separates measured optimizer-loop time from
observed wall-clock spans. The latter include unknown startup, checkpoint,
upload, and idle time; neither is inflated into an unsupported “training time”
claim.

| Segment | Steps counted | Sum of retained `sec_per_step` | Observed timestamp span | Evidence / confidence |
| --- | ---: | ---: | ---: | --- |
| initial production | 1–3000 | 44,620.256 s = **12:23:40** | W&B local start `2026-08-31 05:39:13` to `step_003000.pt` mtime `18:03:59.728` = **12:24:47** | metric JSONL, log, mtime; reconstructed |
| cooldown, all | 3001–5000 | 29,840.203 s = **8:17:20** | `2026-08-31 19:23:10` to `2026-09-01 03:41:18.584` = **8:18:09** | metric JSONL, log, mtime; reconstructed |
| cooldown through parent-4000 | 3001–4000 | 15,015.827 s = **4:10:16** | `19:23:10` to parent mtime `23:33:52.080` = **4:10:42** | endpoint-relevant slice; reconstructed |
| finish-control, all | 4001–4500 | 7,613.232 s = **2:06:53** | `2026-09-01 05:28:36` to `07:35:54.914` = **2:07:19** | metric JSONL, log, mtime; reconstructed |
| finish-control through A4300 | 4001–4300 | 4,628.814 s = **1:17:09** | `05:28:36` to A4300 mtime `06:46:01.393` = **1:17:25** | endpoint-relevant slice; reconstructed |

Endpoint-relevant metric-step total is **64,264.897 seconds = 17:51:05**:
0–3000 plus 3001–4000 plus 4001–4300. The full initial-to-A4300 calendar
interval is **25:06:48**, but contains an approximately 1:19 gap after step
3000 and approximately 5:55 gap after parent-4000.

`sec_per_step` is recorded immediately after the optimizer update. The source
then writes JSONL/W&B and may serialize/mirror a checkpoint; it therefore
measures each optimizer loop but excludes later checkpoint work in that
iteration. The observed checkpoint span bounds operational run time more
closely, but starts after process initialization because the first retained
timestamp is W&B’s local run directory.

## Cache/preprocessing

The complete full-train 768 latent cache has a retained artifact-write window
of **40:21**, from `train_manifest_identity.json` at `2026-08-30 22:14:38.541`
to pose-sidecar metadata at `22:54:59.798`. Its latent shard sequence itself
runs from `22:15:02.795` to `22:54:10.640`. The persistent train text cache
has retained shard writes from `2026-08-26 12:55:27.870` to `13:09:04.857`
(**13:37**). These are lower-bound artifact intervals, not complete process
runtimes: no retained process-start logs prove setup time or absence of gaps.

## Evaluation

Per-stage evaluation summaries exist at `2026-08-31 18:31:25` (initial),
`2026-09-01 05:03:14` (cooldown), and `2026-09-01 10:54:59`
(finish-control), but their retained directories have no start markers, so no
duration is claimed.

The final-val artifact set begins with parent preflight at `2026-09-03 07:09:08.916`
and ends with the retained mix-075 summary at `08:44:56.511`: an observed
artifact span of **1:35:48**. It covers parent, finish-control, and three
interpolation candidates, not just released mix-025. Mix-025-specific retained
files span `08:28:26.490` to `08:44:19.822` (**15:53**), again only a
lower bound on that evaluator segment.

Sources: the three NFS `checkpoints/<run>/metrics.jsonl` files; corresponding
`production-logs/*.log`; NFS checkpoint/cache/evaluation artifact mtimes. Exact
paths and source locators are indexed in
[TRAINING_METHODS_INFRA.md](TRAINING_METHODS_INFRA.md).
