# Hero-v2 final selection

This document freezes the seven new controls selected for the `hero-v2`
showcase addition. It is a generation plan only: no final images are generated
or selected here. The five accepted `hero-v1` winners and all frozen evaluation
artifacts remain unchanged.

## Generation defaults

- Release candidate: `mix-025`
- Mode: `turbo-pose-control` (turbo)
- Geometry: native aspect-preserving dimensions recorded below
- Control scale: `1.0`

The two duo concepts may make one fallback retry at control scale `1.25` only
when the initial `1.0` generation has weak pose adherence. No other scale
change is authorized by this selection.

## Frozen controls and prompts

| # | Concept | Source stem | Final rendered control | Dimensions | People | Fixed seed |
| ---: | --- | --- | --- | --- | ---: | ---: |
| 1 | Realistic female warrior | `real_human_humanart_15000000000016` | `docs/showcase/final/hero-v2/control-audition/controls/realistic-female-warrior/02_real_human_humanart_15000000000016.png` | 896 x 1152 | 1 | 7194308401 |
| 2 | Elegant male warrior / wandering knight | `painting_humanart_9000000000455` | `docs/showcase/final/hero-v2/control-audition/controls/elegant-male-warrior-wandering-knight/01_painting_humanart_9000000000455.png` | 832 x 1216 | 1 | 7194308402 |
| 3 | Moonlit priestess / dreamy floral oracle | `painting_humanart_9000000000724` | `docs/showcase/final/hero-v2/control-audition/controls/moonlit-priestess-dreamy-floral-oracle/02_painting_humanart_9000000000724.png` | 768 x 1344 | 1 | 7194308403 |
| 4 | Stained-glass saint / celestial figure | `sculpture_humanart_14000000004082` | `docs/showcase/final/hero-v2/control-audition/controls/stained-glass-saint-celestial-figure/02_sculpture_humanart_14000000004082.png` | 1024 x 1024 | 1 | 7194308404 |
| 5 | Realistic fashion/editorial portrait | `painting_humanart_9000000001986` | `docs/showcase/final/hero-v2/control-audition/controls/realistic-fashion-editorial-portrait/03_painting_humanart_9000000001986.png` | 896 x 1152 | 1 | 7194308405 |
| 6 | Gothic masked noble with attendant | `real_human_humanart_15000000002158` | `docs/showcase/final/hero-v2/control-audition/controls/gothic-masked-noble-with-attendant/02_real_human_humanart_15000000002158.png` | 1152 x 896 | 2 | 7194308406 |
| 7 | Painterly mythic companions | `painting_humanart_9000000000976` | `docs/showcase/final/hero-v2/control-audition/controls/painterly-mythic-companions/02_painting_humanart_9000000000976.png` | 1216 x 832 | 2 | 7194308407 |

### 1. Realistic female warrior

**Rationale:** Clear upright body and separated arms provide a readable heroic
silhouette while adding grounded cinematic realism to the showcase.

**Prompt:** `A battle-worn female warrior in practical layered steel and dark leather armor, weathered wool cloak and subtle travel gear, on a windswept highland at dawn, natural skin texture, cool mist and restrained golden light, cinematic fantasy realism, tactile materials and atmospheric detail.`

### 2. Elegant male warrior / wandering knight

**Rationale:** The tall, relaxed silhouette is distinct from the existing mage
and supports refined travel clothing without prescribing body geometry.

**Prompt:** `An elegant wandering knight in tailored charcoal wool, aged steel accents and a deep green travel cloak, crossing a quiet stone moor beneath an overcast sky, soft directional daylight, refined realistic fantasy photography, natural fabric texture and muted cinematic color.`

### 3. Moonlit priestess / dreamy floral oracle

**Rationale:** The asymmetric stance and separated hands support a calm oracle
reading, adding a soft floral and silvery-blue register.

**Prompt:** `A moonlit floral oracle in layered ivory silk and pale blue embroidered veils, surrounded by night-blooming flowers and drifting petals in a secluded garden, silver lunar glow, dreamy romantic fantasy painting, delicate textures and luminous pastel color.`

### 4. Stained-glass saint / celestial figure

**Rationale:** The symmetric near-square control is suited to a graphic sacred
icon treatment and expands the set with a luminous celestial image.

**Prompt:** `A celestial saint in radiant ivory and jewel-toned ceremonial robes, before an ornate cathedral of stained glass and gilded tracery, prismatic light and a quiet sacred atmosphere, richly detailed luminous sacred-art illustration, deep sapphire, ruby and gold.`

### 5. Realistic fashion/editorial portrait

**Rationale:** The asymmetric upright geometry supports a polished editorial
look while remaining visually separate from the comic-fashion hero.

**Prompt:** `A contemporary fashion model in a sculptural graphite tailored ensemble with brushed-metal jewelry, within a minimalist terracotta studio, soft sculpting daylight and subtle film grain, premium photographic fashion editorial, natural skin texture and restrained modern color.`

### 6. Gothic masked noble with attendant

**Rationale:** Two upright, separated full-body figures provide an intentional,
readable gothic duo with low limb overlap.

**Prompt:** `A masked gothic noble and one devoted attendant in black velvet, antique silver and muted crimson ceremonial dress, inside a candlelit ancestral hall of dark stone and faded portraits, low amber light and deep shadows, elegant cinematic dark-fantasy painting, ornate textile detail.`

Fallback only: retry at control scale `1.25` if the initial scale-`1.0`
generation has weak two-person adherence.

### 7. Painterly mythic companions

**Rationale:** The two readable walking-scale silhouettes form a coherent pair
with low overlap and add a warm landscape-oriented painterly duo.

**Prompt:** `Two mythic companions in weathered indigo, ochre and moss-green travel layers, moving through an ancient valley of tall grasses and distant standing stones, warm late-afternoon light, expressive painterly fantasy illustration, textured brushwork and a quiet sense of shared journey.`

Fallback only: retry at control scale `1.25` if the initial scale-`1.0`
generation has weak two-person adherence.

## Execution contract

For each concept, generate at the recorded dimensions with its frozen control,
fixed seed, `mix-025`, turbo mode, and control scale `1.0`. Treat the control
as the sole source of body geometry; do not change person count, framing, or
limb placement through prompting. Save outputs as a new hero-v2 generation
batch; do not overwrite the audition controls, `hero-v1`, or evaluation
artifacts.
