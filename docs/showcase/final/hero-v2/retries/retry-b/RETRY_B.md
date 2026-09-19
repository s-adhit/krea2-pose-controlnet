# Hero-v2 retry-b plan

This batch contains exactly three new retry-b concepts. It does not modify or
write under `canonical-v1`, `retry-a`, accepted keepers, or earlier batches.

| Concept | Control | Native size | Seed | Scale |
| --- | --- | ---: | ---: | ---: |
| realistic female warrior | `03_real_human_humanart_15000000001026.png` | 1216 x 832 | 7194308601 | 1.25 |
| moonlit lotus princess | `02_painting_humanart_9000000000724.png` | 768 x 1344 | 7194308602 | 1.0 |
| astral empress / cosmic oracle | `02_sculpture_humanart_14000000004082.png` | 1024 x 1024 | 7194308603 | 1.0 |

All controls retain their native geometry. Runtime is `mix-025`, Krea-2 Turbo,
8 steps, CFG 0, mu 1.15, resolution-dependent mu disabled, and no Style-LoRA.

NFS outputs are planned for
`/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/retry-b/`. Presentation copies,
sidecars, copied controls, and the contact sheet are planned for
`docs/showcase/final/hero-v2/generations/retry-b/`. The runner rejects
differing existing output files rather than overwriting them.

Run on the writable GH200 host:

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/generate_hero_v2_retry.py preflight --batch retry-b
uv run python scripts/generate_hero_v2_retry.py generate --batch retry-b
```
