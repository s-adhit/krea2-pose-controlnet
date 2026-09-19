# Frozen ablation and stress results

Source: `docs/blog-evidence/EVALUATION_RESULTS.md`. Values retain frozen payload precision.

## Native vs dynamic-768

| Geometry mode | Attempted / evaluated | Reference people | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native cached geometry | 5 / 5 | 14 | 14 | 0.2920353982 | 0.4026548673 | 0.5884955752 | 0.3074624562 |
| Dynamic-768 bucket | 5 / 5 | 14 | 14 | 0.2477876106 | 0.3584070796 | 0.5309734513 | 0.3096808381 |

Geometry/control-input-path ablation, not a pure resolution-only experiment.

## Hard single-person vs hard multi-person

| Subset | Attempted / evaluated | Reference people | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard single-person | 8 / 8 | 8 | 8 | 0.3727272727 | 0.5181818182 | 0.6818181818 | 0.3281741025 |
| Hard multi-person | 4 / 4 | 12 | 12 | 0.2916666667 | 0.4097222222 | 0.5625000000 | 0.3174141528 |

## Prompt injection

| Condition | Attempted / evaluated | Reference people | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal mix-025 final-val | 48 / 48 | 101 | 95 | 0.4520725389 | 0.6042746114 | 0.7240932642 | 0.3369378586 |
| mix-025 injected prompt | 48 / 48 | 101 | 96 | 0.4009067358 | 0.5563471503 | 0.6832901554 | 0.3394259131 |

Prompt-injection evaluation uses the same frozen final-val stems; prompt text changed while control and evaluation setup remained fixed. CLIP uses the actual injected prompt in the injected condition.
