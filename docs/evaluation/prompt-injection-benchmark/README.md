# Prompt-injection / cross-domain pose benchmark

This benchmark tests whether geometry is controlled by the pose condition while
appearance and semantics can be replaced through text.

It is separate from the frozen source-caption final validation benchmark.

## Prompt-injection 48

`prompt_injection_48.jsonl` maps the same 48 frozen held-out pose conditions to
new human-authored prompts.

Prompt families are balanced:

- 12 photorealistic
- 12 illustration
- 12 material/domain transformation
- 12 cinematic/fantasy

The prompts intentionally avoid explicit body-pose instructions where possible.
Subject count may be specified, but limb configuration is expected to come from
the pose condition.

The set deliberately mixes domains, including:

- photographic pose -> illustration/sculpture/fantasy
- painting pose -> photography/material/cinematic
- real-human pose -> illustration/sculpture/fantasy
- sculpture pose -> photography/illustration/living-subject interpretation

Evaluation should use:

- the same frozen pose controls
- deterministic seeds
- native/aspect-preserving geometry
- Turbo 8 steps
- CFG 0
- mu 1.15
- control scale 1.0
- PCK against the pose condition
- CLIP against the injected prompt

## Same-pose hero

`hero_same_pose_6.jsonl` holds one canonical pose constant while changing the
prompt across six report-safe interpretations:

1. fantasy mage / stained glass
2. modern sorcerer / street mural
3. fashion warrior / comic ink
4. realistic human
5. bronze sculpture
6. painterly interpretation

This set is intended for the README/blog hero figure and for demonstrating the
recommended prompting pattern: describe appearance, material, style, lighting,
and environment while allowing the control image to specify body geometry.
