# Appendix v1

Twelve validated project controls × four short, geometry-neutral appearance
families (48 planned generations). The skeleton is the sole body-geometry
source; prompts specify only visual treatment. The frozen runtime is Krea-2
Turbo, 8 steps, CFG 0, mu 1.15, control scale 1.0, no Style-LoRA, and
native/aspect-preserving geometry selected with
`choose_bucket(source_size, REFERENCE_KREA_BUCKETS)`.

`manifest.json` is the source of reproducibility facts.  A `pending_cuda`
record has been validated but requires execution from a CUDA-visible GH200
shell; it is not an implicit retry or a completed generation. Once all rows
succeed, `contact_sheet.png` has one row per control and five image columns:
`control`, then the four style families. Images are letterboxed, never cropped.
