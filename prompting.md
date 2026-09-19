# Prompting Krea-2 Pose Control

## Core rule

**The pose control determines broad geometry; the prompt determines compatible appearance and rendering.** Use the skeleton for body placement and text for the subject, clothing, environment, lighting, and style. Hands and fingers are not directly controlled.

## Reusable recipe

```text
[subject/archetype], [identity/clothing/material], [environment], [lighting],
[rendering/style], [optional broad pose-compatible mood]
```

Do not narrate joints one by one. Avoid text that conflicts with the control's pose, framing, or person count. If the intended body configuration changes, change the pose control rather than trying to correct it in the prompt.

## Control selection and scale

Use clean, readable controls for showcase work. Start at control scale `1.0`; `1.25–1.5` is useful when stronger adherence is needed, while `<=0.5` is weak. The frozen examples below cover realistic/fantasy, painterly, comic/fashion, gothic, celestial/floral, and duo scenes.

## Frozen final examples

The ordered final winner contract is [hero-v2/final_winners.json](docs/showcase/final/hero-v2/final_winners.json). Every example uses candidate `mix-025`.

### 01. Psychedelic swordswoman

<img src="docs/showcase/final/hero-v2/final/01_psychedelic-swordswoman_condition.png" alt="Psychedelic swordswoman condition" width="180"> <img src="docs/showcase/final/hero-v2/final/01_psychedelic-swordswoman_generation.png" alt="Psychedelic swordswoman generation" width="180">

> A female swordswoman in ornate fantasy clothing wielding a long glowing sword, stylized digital painting with bold expressive brushstrokes and a vibrant psychedelic color palette, iridescent marbled fabrics, surreal fantasy mood, deep shadows, dramatic theatrical lighting, highly detailed concept art.

Seed `7194308222` · Native `896 × 1152` · Candidate `mix-025` · Control scale `1.0`

### 02. Fantasy mage

<img src="docs/showcase/final/hero-v2/final/02_fantasy-mage_condition.png" alt="Fantasy mage condition" width="180"> <img src="docs/showcase/final/hero-v2/final/02_fantasy-mage_generation.png" alt="Fantasy mage generation" width="180">

> an ancient fantasy mage in layered sapphire, burgundy and ivory ceremonial robes, intricate gold embroidery, arcane jewelry and weathered magical textiles, inside a vast ruined cathedral filled with drifting dust and fragments of glowing runes, cold blue window light mixed with warm candlelight, richly detailed cinematic fantasy painting, dramatic atmosphere, tactile fabrics and ornate fantasy craftsmanship

Seed `1847302951` · Native `896 × 1152` · Candidate `mix-025` · Control scale `1.0`

### 03. Comic fashion

<img src="docs/showcase/final/hero-v2/final/03_comic-fashion_condition.png" alt="Comic fashion condition" width="240"> <img src="docs/showcase/final/hero-v2/final/03_comic-fashion_generation.png" alt="Comic fashion generation" width="240">

> a severe dark-haired comic-fashion antihero wearing a long structured charcoal coat, sharp tailored layers, metallic accessories and bold geometric details, photographed like an avant-garde fashion editorial against graphic architectural shapes, dramatic hard shadows, deep indigo and black palette with selective gold accents, expressive manga-inspired linework blended with high-fashion photography, powerful cinematic composition

Seed `2519074836` · Native `1216 × 832` · Candidate `mix-025` · Control scale `1.0`

### 04. Starry-night painterly

<img src="docs/showcase/final/hero-v2/final/04_starry-night-painterly_condition.png" alt="Starry-night painterly condition" width="130"> <img src="docs/showcase/final/hero-v2/final/04_starry-night-painterly_generation.png" alt="Starry-night painterly generation" width="130">

> A solitary figure in flowing clothing beneath a swirling star-filled night sky, painterly dreamlike scene with expressive brushwork, luminous blues and warm gold highlights, poetic atmosphere, richly textured, highly detailed, fantasy-inspired nightscape.

Seed `7194308251` · Native `704 × 1472` · Candidate `mix-025` · Control scale `1.0`

### 05. Dark-fantasy jester

<img src="docs/showcase/final/hero-v2/final/05_dark-fantasy-jester_condition.png" alt="Dark-fantasy jester condition" width="260"> <img src="docs/showcase/final/hero-v2/final/05_dark-fantasy-jester_generation.png" alt="Dark-fantasy jester generation" width="260">

> a sinister dark-fantasy court jester in elaborate black, crimson and antique-gold ceremonial clothing, asymmetric silk panels, embroidered bells, dramatic ruffled collar, intricate harlequin details and a mysterious theatrical mask, inside a decaying gothic palace hall with faded murals and candlelit stone arches, deep shadows with saturated jewel-tone highlights, painterly dark-fantasy concept art, elegant, strange and unsettling

Seed `3028147759` · Native `1472 × 704` · Candidate `mix-025` · Control scale `1.0`

### 06. Elegant male warrior / wandering knight

<img src="docs/showcase/final/hero-v2/final/06_elegant-male-warrior-wandering-knight_condition.png" alt="Wandering knight condition" width="160"> <img src="docs/showcase/final/hero-v2/final/06_elegant-male-warrior-wandering-knight_generation.png" alt="Wandering knight generation" width="160">

> An elegant wandering knight in tailored charcoal wool, aged steel accents and a deep green travel cloak, crossing a quiet stone moor beneath an overcast sky, soft directional daylight, refined realistic fantasy photography, natural fabric texture and muted cinematic color.

Seed `7194308402` · Native `832 × 1216` · Candidate `mix-025` · Control scale `1.0`

### 07. Gothic masked noble with attendant

<img src="docs/showcase/final/hero-v2/final/07_gothic-masked-noble-with-attendant_condition.png" alt="Gothic noble duo condition" width="240"> <img src="docs/showcase/final/hero-v2/final/07_gothic-masked-noble-with-attendant_generation.png" alt="Gothic noble duo generation" width="240">

> A masked gothic noble and one devoted attendant in black velvet, antique silver and muted crimson ceremonial dress, inside a candlelit ancestral hall of dark stone and faded portraits, low amber light and deep shadows, elegant cinematic dark-fantasy painting, ornate textile detail.

Seed `7194308406` · Native `1152 × 896` · Candidate `mix-025` · Control scale `1.0`

### 08. Painterly mythic companions

<img src="docs/showcase/final/hero-v2/final/08_painterly-mythic-companions_condition.png" alt="Mythic companions condition" width="240"> <img src="docs/showcase/final/hero-v2/final/08_painterly-mythic-companions_generation.png" alt="Mythic companions generation" width="240">

> Exactly two mythic companions, both fully visible, in weathered indigo, ochre and moss-green travel layers, moving together through an ancient valley of tall grasses and distant standing stones, warm late-afternoon light, expressive painterly fantasy illustration, textured brushwork, and no additional people.

Seed `7194308505` · Native `1216 × 832` · Candidate `mix-025` · Control scale `1.25`

### 09. Moonlit lotus princess

<img src="docs/showcase/final/hero-v2/final/09_moonlit-lotus-princess_condition.png" alt="Moonlit lotus princess condition" width="140"> <img src="docs/showcase/final/hero-v2/final/09_moonlit-lotus-princess_generation.png" alt="Moonlit lotus princess generation" width="140">

> A moonlit lotus princess in flowing ivory and pale blue silk robes, seated peacefully among blooming night flowers and floating petals in an enchanted garden, silver lunar glow, dreamy romantic fantasy painting, delicate textures, luminous floral atmosphere, and elegant serenity.

Seed `7194308602` · Native `768 × 1344` · Candidate `mix-025` · Control scale `1.0`

### 10. Astral empress / cosmic oracle

<img src="docs/showcase/final/hero-v2/final/10_astral-empress-cosmic-oracle_condition.png" alt="Astral empress condition" width="200"> <img src="docs/showcase/final/hero-v2/final/10_astral-empress-cosmic-oracle_generation.png" alt="Astral empress generation" width="200">

> A cosmic oracle seated in a centered meditative pose, wearing radiant midnight-blue, ivory, and gold ceremonial robes, surrounded by celestial motifs, star maps, luminous halos, and prismatic sacred geometry, in a surreal astral sanctuary, richly detailed mystical illustration, luminous atmosphere, elegant symmetry, and deep cosmic color.

Seed `7194308603` · Native `1024 × 1024` · Candidate `mix-025` · Control scale `1.0`

## A conflicting prompt to avoid

This is the exact `P5_conflicting` inversion example from the existing [prompting study](docs/evaluation/prompting-guide/prompting_study.jsonl):

> A single adult woman standing straight with both feet on the floor, wearing a fitted dark violet athletic outfit, realistic studio photography.

It conflicts with an inverted control. Likewise, do not ask for a close-up from a full-body control or change a one-person control into a multi-person scene.

## Pasteable LLM instruction

```text
Write one Krea-2 Pose Control prompt for the supplied pose condition and concept.
Treat the pose image as the source of body geometry. Describe only compatible
subject identity, clothing/materials, environment, lighting, rendering/style,
and at most one broad pose-compatible mood. Do not narrate joints, prescribe
hand/finger geometry, change subject count, or request conflicting framing.
Return only the prompt.
```

## Optional Style-LoRA guidance

Style-LoRA is a separate optional composition choice, not part of the final hero workflow. When using one, keep its official trigger and strength explicit, introduce strong style language gradually, and keep the pose prompt otherwise geometry-neutral.
