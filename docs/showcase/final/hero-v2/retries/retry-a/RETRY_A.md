# Hero-v2 retry-a plan

This selective batch retries five non-keeper `canonical-v1` concepts. The two
accepted canonical-v1 keepers are excluded. Neither this plan nor its runner
writes under `canonical-v1`.

| Concept | Control | Size | Seed | Scale |
| --- | --- | ---: | ---: | ---: |
| realistic female warrior | replacement audition control `03_real_human_humanart_15000000001026.png` | 1216 x 832 | 7194308501 | 1.0 |
| moonlit priestess / dreamy floral oracle | frozen canonical control | 768 x 1344 | 7194308502 | 1.0 |
| stained-glass saint / celestial figure | frozen canonical control | 1024 x 1024 | 7194308503 | 1.0 |
| realistic fashion/editorial portrait | frozen canonical control | 896 x 1152 | 7194308504 | 1.0 |
| painterly mythic companions | frozen canonical control | 1216 x 832 | 7194308505 | 1.25 |

The warrior replacement control is landscape 1216 x 832; its retry therefore
uses that control's native geometry. All other controls retain their frozen
canonical geometry. Runtime remains `mix-025`, Krea-2 Turbo, 8 steps, CFG 0,
mu 1.15, with resolution-dependent mu disabled and no Style-LoRA.

Outputs are planned for
`/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/retry-a/`; presentation copies,
sidecars, and a contact sheet go to
`docs/showcase/final/hero-v2/generations/retry-a/`. Existing differing files
are treated as errors rather than overwritten.

Run on the writable GH200 host:

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/generate_hero_v2_retry.py preflight
uv run python scripts/generate_hero_v2_retry.py generate
```
