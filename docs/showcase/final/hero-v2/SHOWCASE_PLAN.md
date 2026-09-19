# Hero v2 final-showcase expansion plan

## Scope and guardrails

This is a planning-only audit. It inventories the five accepted `hero-v1`
winners and defines a 12-image target for the next final showcase. It does not
generate an image, select a new final control, or modify any frozen evaluation
or hero-v1 artifact. `hero-v1/final_winners.json` remains the provenance source
for the retained images.

The public presentation rule remains: the control supplies body geometry;
prompting supplies identity, clothes, setting, lighting, and rendering. Use
clean readable single-person controls by default and control scale `1.0`; only
audition `1.25–1.5` where a composition demonstrably benefits from firmer
adherence.

## Accepted hero-v1 inventory

All five accepted winners are single-person images, use release candidate
`mix-025`, and were sampled in `turbo-pose-control` mode at control scale
`1.0`. Dimensions are the matched control/output dimensions.

| Concept / accepted generation | Count | Control image | Generated image | Prompt | Seed | Dimensions | Candidate | Scale |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| Psychedelic swordswoman / `female_swordswoman_psychedelic_original` | Single | `docs/showcase/final/hero-v1/conditions/female_swordswoman_psychedelic.png` | `docs/showcase/final/hero-v1/original_winners/female_swordswoman_psychedelic.png` | Female swordswoman in ornate fantasy clothing wielding a long glowing sword; bold expressive digital paint, vibrant psychedelic palette, iridescent marbled fabrics, surreal fantasy mood, deep shadows and theatrical light. | 7194308222 | 896 x 1152 | `mix-025` | 1.0 |
| Fantasy mage / `fantasy_mage_hero_b` | Single | `docs/showcase/final/hero-v1/conditions/fantasy_mage.png` | `docs/showcase/final/hero-v1/hero_variants/fantasy_mage_hero_b.png` | Ancient fantasy mage in sapphire, burgundy, and ivory ceremonial robes with gold embroidery and arcane jewelry, in a ruined rune-lit cathedral; richly detailed cinematic fantasy painting. | 7194308302 | 896 x 1152 | `mix-025` | 1.0 |
| Comic fashion / `comic_fashion_hero_b` | Single | `docs/showcase/final/hero-v1/conditions/comic_fashion.png` | `docs/showcase/final/hero-v1/hero_variants/comic_fashion_hero_b.png` | Severe dark-haired comic-fashion antihero in a structured charcoal coat and metallic details, against graphic architecture; hard-shadowed indigo/black/gold avant-garde fashion editorial with manga linework. | 7194308322 | 1216 x 832 | `mix-025` | 1.0 |
| Starry-night painterly / `starry_night_painterly_hero_a` | Single | `docs/showcase/final/hero-v1/conditions/starry_night_painterly.png` | `docs/showcase/final/hero-v1/hero_variants/starry_night_painterly_hero_a.png` | Solitary flowing figure beneath a swirling star-filled sky; dreamlike expressive brushwork, luminous blue and warm gold, poetic textured fantasy nightscape. | 7194308341 | 704 x 1472 | `mix-025` | 1.0 |
| Dark-fantasy jester / `dark_fantasy_jester_original` | Single | `docs/showcase/final/hero-v1/conditions/dark_fantasy_jester.png` | `docs/showcase/final/hero-v1/original_winners/dark_fantasy_jester.png` | Sinister masked court jester in black, crimson, and antique gold, with bells, ruff, and harlequin detail in a decaying candlelit gothic palace; elegant unsettling dark-fantasy concept art. | 3028147759 | 1472 x 704 | `mix-025` | 1.0 |

### Exact accepted prompts

These are the exact prompt strings recorded for the selected winners:

- **Psychedelic swordswoman:** `A female swordswoman in ornate fantasy clothing wielding a long glowing sword, stylized digital painting with bold expressive brushstrokes and a vibrant psychedelic color palette, iridescent marbled fabrics, surreal fantasy mood, deep shadows, dramatic theatrical lighting, highly detailed concept art.`
- **Fantasy mage:** `an ancient fantasy mage in layered sapphire, burgundy and ivory ceremonial robes, intricate gold embroidery, arcane jewelry and weathered magical textiles, inside a vast ruined cathedral filled with drifting dust and fragments of glowing runes, cold blue window light mixed with warm candlelight, richly detailed cinematic fantasy painting, dramatic atmosphere, tactile fabrics and ornate fantasy craftsmanship`
- **Comic fashion:** `a severe dark-haired comic-fashion antihero wearing a long structured charcoal coat, sharp tailored layers, metallic accessories and bold geometric details, photographed like an avant-garde fashion editorial against graphic architectural shapes, dramatic hard shadows, deep indigo and black palette with selective gold accents, expressive manga-inspired linework blended with high-fashion photography, powerful cinematic composition`
- **Starry-night painterly:** `A solitary figure in flowing clothing beneath a swirling star-filled night sky, painterly dreamlike scene with expressive brushwork, luminous blues and warm gold highlights, poetic atmosphere, richly textured, highly detailed, fantasy-inspired nightscape.`
- **Dark-fantasy jester:** `a sinister dark-fantasy court jester in elaborate black, crimson and antique-gold ceremonial clothing, asymmetric silk panels, embroidered bells, dramatic ruffled collar, intricate harlequin details and a mysterious theatrical mask, inside a decaying gothic palace hall with faded murals and candlelit stone arches, deep shadows with saturated jewel-tone highlights, painterly dark-fantasy concept art, elegant, strange and unsettling`

The frozen winner contract and selected sidecars remain the authoritative
provenance source.

## Retention decision

Retain all five accepted `hero-v1` winners in the expanded set. Each has a
clear readable pose, a coherent prompt-to-image match, and a distinct hero use:

| Retain | Reason in v2 |
| --- | --- |
| Psychedelic swordswoman | The strongest vivid action image; proves a dynamic prop-compatible fantasy pose. |
| Fantasy mage | The most legible cinematic fantasy/environment image; anchors the painterly fantasy category. |
| Comic fashion | The clearest editorial/comic crossover and a useful landscape counterweight. |
| Starry-night painterly | Adds quiet vertical composition and a recognizably painterly visual language. |
| Dark-fantasy jester | The most theatrical gothic image; preserves a dark, characterful wide composition. |

They should be copied or linked as already-approved presentation assets, not
regenerated. Their frozen sidecars, controls, and winner contract must remain
unchanged.

## Gaps to close

The current five lean heavily toward illustrated fantasy: four are primarily
fantasy/painterly and there is no grounded photographic hero, no clear
light/celestial counterpart to the jester, no floral/romantic palette, and no
intentional two-person proof. Presentation is also skewed to one masculine
elder mage and otherwise stylized figures. The v2 addition should create:

- realistic female and male leads with natural skin, wardrobe, and lighting;
- one clean editorial photograph distinct from comic-fashion illustration;
- luminous celestial/stained-glass and soft floral registers;
- a restrained pair of readable duos, rather than a crowd or tangled control;
- a balance of portrait, near-square, and landscape geometry.

## Proposed 12-image final target

Target composition: **10 single-person images and 2 duo images**. The five
retained assets remain exactly as accepted; the seven marked **new** are
concept/control auditions, not generation authorization.

| # | Concept | Count | Status | Style / purpose | Control audition brief |
| ---: | --- | --- | --- | --- | --- |
| 1 | Psychedelic swordswoman | Single | Retain | Saturated action fantasy / digital painting | Preserve approved vertical sword control. |
| 2 | Fantasy mage | Single | Retain | Cinematic fantasy / cathedral painterly | Preserve approved vertical mage control. |
| 3 | Comic fashion antihero | Single | Retain | Comic-fashion / editorial hybrid | Preserve approved wide reclining control. |
| 4 | Starry-night wanderer | Single | Retain | Poetic painterly / nocturne | Preserve approved tall standing control. |
| 5 | Dark-fantasy jester | Single | Retain | Gothic theatrical / dark fantasy | Preserve approved wide gestural control. |
| 6 | Realistic female warrior | Single | New | Grounded cinematic realism: practical armor, weathered landscape, natural skin and material detail | Full-body, readable weapon-side silhouette; avoid crossed limbs and tiny head. |
| 7 | Elegant male warrior / wandering knight | Single | New | Refined realistic fantasy: tailored travel layers, steel and wool, overcast or golden-hour setting | Upright or walking three-quarter pose, visually distinct from mage and comic antihero. |
| 8 | Moonlit priestess / dreamy floral oracle | Single | New | Moonlit celestial romance with pale florals, silvery blue, and a soft luminous palette | Open-arm or calm standing portrait; hands separated from torso. This deliberately combines the moonlit oracle and dreamy-floral gap. |
| 9 | Stained-glass saint / celestial figure | Single | New | Luminous graphic sacred-art treatment: jewel glass, halo-like backlight, strong silhouette | Centered near-square or portrait control with clean arm separation; do not reuse the mage’s exact control. |
| 10 | Realistic fashion/editorial portrait | Single | New | Premium photographic editorial: sculptural wardrobe, studio color field, natural texture | Full/three-quarter body fashion stance with clean negative space; explicitly non-comic rendering. |
| 11 | Gothic masked noble with attendant | Duo | New | Controlled dark-gothic pair: masked noble and one attendant in an architectural interior | Exactly two well-separated, similarly scaled people; readable interaction, no overlap, no crowd. |
| 12 | Painterly mythic companions | Duo | New | Warm, expressive mythic painting: two travelers or guardians in a single coherent scene | Exactly two full bodies with distinct silhouettes; landscape composition and no occluded limbs. |

This lineup covers realistic, fantasy, painterly, fashion/editorial,
gothic, and celestial styles while containing duo risk to two final images.
The two duo concepts should be the last images admitted: retain a single-person
fallback for each until person count, separation, and composition are verified.

## Control-audition next step

Create a new **control-only** audition; do not render final prompts during this
step. The existing `pose-audition-v5` controls are useful one-person starting
references, while `pose-audition-v4` demonstrates why multi-person candidates
must be screened carefully. For each proposed new concept, collect 3–5 native
aspect-preserving candidate controls and record source stem, authoritative
person count, bucket, and rendered-control dimensions.

Accept a single-person candidate only if the skeleton has one readable body,
clear torso/head placement, separated limbs, and useful compositional space.
Accept a duo candidate only if it has exactly two readable people, distinct
heads/torso boxes, low limb overlap, compatible scale, and no incidental third
person. Reject crowd controls and controls whose apparent pose requires
finger-level fidelity. Keep the control image and eventual generation at the
same native bucket geometry.

After controls are selected, write the prompt set using `prompting.md`:
geometry-neutral appearance/style language, no joint narration, no count or
framing conflict. Sample at control scale `1.0` first; only test a higher scale
for a documented adherence failure. No frozen `hero-v1` or evaluation file is
an input to overwrite.

## Sources audited

- `docs/showcase/final/hero-v1/` and `final_winners.json` (accepted images,
  exact selected sidecars, native dimensions, candidate, seeds, and scales)
- `README.md` and `prompting.md` (public showcase and prompting contract)
- `/lambda/nfs/adhit/krea2-pose/showcase` (earlier final batches, hero variants,
  and pose-audition controls)
- `/lambda/nfs/adhit/krea2-pose/evaluation/hero-same-pose` (evidence that the
  current candidate supports distinct photoreal, painterly, stained-glass, and
  comic prompt regimes; its outputs are evaluation artifacts and are not being
  promoted or modified here)
