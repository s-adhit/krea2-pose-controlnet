# Prompting Krea-2 Pose Control

## Core rule

**The pose control determines geometry; the prompt determines compatible appearance and rendering.** The skeleton establishes the body’s broad structure and placement. Use text for who the subject is, what they wear, where they are, and how the image looks.

The control has body keypoints only. It does **not** provide finger-level control or facial keypoints.

## Reusable recipe

```text
[subject/archetype], [identity/clothing/material], [environment], [lighting],
[rendering/style], [optional broad pose-compatible mood]
```

For example, add a broad phrase such as “dynamic airborne moment” only when it helps explain the existing control. Do not narrate joints one by one.

Avoid:

- detailed limb or joint narration;
- instructions that conflict with the control’s limb placement, subject count, or framing;
- explicit hand geometry as a supposed anatomy fix.

If the intended body configuration changes, change the pose control rather than trying to correct it in the prompt.

## Control selection and scale

Use clean, readable single-person controls for public/showcase work. Use control scale `1.0` by default; `1.25–1.5` is an explicit option when stronger adherence is useful, while `<=0.5` is weak. Inversion, compressed/tangled, and multiperson controls remain useful for evaluation, but were less reliable aesthetically in final hero selection.

## Frozen final examples

These five public winners use the following exact frozen prompts; their images, conditions, seeds, and sidecars are recorded in [the winner contract](docs/showcase/final/hero-v1/final_winners.json).

### Fantasy mage

> an ancient fantasy mage in layered sapphire, burgundy and ivory ceremonial robes, intricate gold embroidery, arcane jewelry and weathered magical textiles, inside a vast ruined cathedral filled with drifting dust and fragments of glowing runes, cold blue window light mixed with warm candlelight, richly detailed cinematic fantasy painting, dramatic atmosphere, tactile fabrics and ornate fantasy craftsmanship

### Dark-fantasy jester

> a sinister dark-fantasy court jester in elaborate black, crimson and antique-gold ceremonial clothing, asymmetric silk panels, embroidered bells, dramatic ruffled collar, intricate harlequin details and a mysterious theatrical mask, inside a decaying gothic palace hall with faded murals and candlelit stone arches, deep shadows with saturated jewel-tone highlights, painterly dark-fantasy concept art, elegant, strange and unsettling

### Comic fashion

> a severe dark-haired comic-fashion antihero wearing a long structured charcoal coat, sharp tailored layers, metallic accessories and bold geometric details, photographed like an avant-garde fashion editorial against graphic architectural shapes, dramatic hard shadows, deep indigo and black palette with selective gold accents, expressive manga-inspired linework blended with high-fashion photography, powerful cinematic composition

### Psychedelic swordswoman

> A female swordswoman in ornate fantasy clothing wielding a long glowing sword, stylized digital painting with bold expressive brushstrokes and a vibrant psychedelic color palette, iridescent marbled fabrics, surreal fantasy mood, deep shadows, dramatic theatrical lighting, highly detailed concept art.

### Starry-night painterly

> A solitary figure in flowing clothing beneath a swirling star-filled night sky, painterly dreamlike scene with expressive brushwork, luminous blues and warm gold highlights, poetic atmosphere, richly textured, highly detailed, fantasy-inspired nightscape.

## A conflicting prompt to avoid

This is the exact `P5_conflicting` inversion example from the existing [prompting study](docs/evaluation/prompting-guide/prompting_study.jsonl):

> A single adult woman standing straight with both feet on the floor, wearing a fitted dark violet athletic outfit, realistic studio photography.

It conflicts with an inverted control. The study’s failure modes also include asking for a close-up from a full-body control or changing a one-person control into a multi-person scene.

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
