# Krea-2 Pose Control-LoRA — frozen evaluation results

Status: direct result-artifact record, audited 2026-09-19 UTC. Values retain the result payload precision shown below. PCK values are globally pooled eligible joints under the methodology in [EVALUATION_METHODS.md](EVALUATION_METHODS.md); CLIP is mean image/prompt cosine similarity. “Attempted” means complete required generations, not a subset selected after scoring.

## Final-val candidate comparison

Frozen manifest: `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json`; 48 attempted/evaluated images, 0 unavailable/failed scoring images, 27 single-person plus 21 multi-person evaluable images, 101 renderer-qualified reference people, and 1,544 eligible joints. Native geometry; Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`; fixed per-stem sampling seeds; control scale 1.0 for pose candidates.

| Candidate | Attempted / evaluated | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turbo baseline | 48 / 48 | 95 | 0.0356217617 | 0.1101036269 | 0.3387305699 | 0.3440154150 |
| parent-4000 | 48 / 48 | 94 | 0.4300518135 | 0.5725388601 | 0.7104922280 | 0.3364916809 |
| mix-025 | 48 / 48 | 95 | 0.4520725389 | 0.6042746114 | 0.7240932642 | 0.3369378586 |
| mix-050 | 48 / 48 | 95 | 0.4443005181 | 0.5990932642 | 0.7130829016 | 0.3341130456 |
| mix-075 | 48 / 48 | 94 | 0.4475388601 | 0.6036269430 | 0.7085492228 | 0.3354207429 |
| A4300 / finish-control-a4300 | 48 / 48 | 93 | 0.4404145078 | 0.5939119171 | 0.7169689119 | 0.3369788169 |

Evidence: NFS final-val `pck_clip_results.json` for each non-base candidate; NFS `turbo-baseline/turbo-baseline-final-val-v1/pck_clip_results.json`, `aggregate_by_candidate.rows`; committed rounded cross-check `docs/evaluation/final-val-turbo/results_summary.json`. **Verified.**

## Native versus dynamic-768

Frozen five-condition specification: `docs/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1.json`; mix-025, native control scale 1.0, Krea-2 Turbo 8/0/`mu=1.15`, same fixed condition seeds. Each mode has 5 attempted/evaluated images and 14 renderer-qualified references / 226 eligible joints; no unavailable score records.

| Geometry mode | Attempted / evaluated | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native cached geometry | 5 / 5 | 14 | 0.2920353982 | 0.4026548673 | 0.5884955752 | 0.3074624562 |
| dynamic-768 bucket | 5 / 5 | 14 | 0.2477876106 | 0.3584070796 | 0.5309734513 | 0.3096808381 |

Evidence: `docs/evaluation/native-vs-dynamic768/results/mix-025-control1-turbo-v1/metrics_by_geometry.json`, `rows`. **Verified.**

## Hard-pose / multi-person stress

Frozen specification: `docs/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1.json`; mix-025, native geometry, control scale 1.0, Krea-2 Turbo 8/0/`mu=1.15`, 12 fixed-seed generations. All 12 were scored; no unavailable records. The frozen design contains 8 single-person and 4 multi-person images.

| Subset | Attempted / evaluated | Reference people | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard single-person | 8 / 8 | 8 | 8 | 0.3727272727 | 0.5181818182 | 0.6818181818 | 0.3281741025 |
| hard multi-person | 4 / 4 | 12 | 12 | 0.2916666667 | 0.4097222222 | 0.5625000000 | 0.3174141528 |

Evidence: `docs/evaluation/hard-pose-multiperson/results/hard-pose-multiperson-mix-025-v1/pck_clip_results.json`, `aggregate_by_person_group.rows`. **Verified.**

## Prompt-injection: mix-025

Frozen prompt manifest: `docs/evaluation/prompt-injection-benchmark/prompt_injection_48.jsonl`; exactly the final-val 48 stems in frozen order, with injected prompts. Mix-025, native geometry, control scale 1.0, Krea-2 Turbo 8/0/`mu=1.15`, fixed final-val sampling seeds. All 48 images were generated and scored; 101 renderer-qualified reference people and 1,544 eligible joints.

| Candidate | Attempted / evaluated | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix-025, injected prompt | 48 / 48 | 96 | 0.4009067358 | 0.5563471503 | 0.6832901554 | 0.3394259131 |

There were 5 unmatched reference people and 92 unmatched generated people; joint-evaluation coverage was `0.9501295337`. CLIP was computed against injected prompt text. Evidence: `docs/evaluation/prompt-injection-benchmark/results/mix-025/pck_clip_results.json`, `checkpoints[0]`; `evaluation_summary.json`, `clip_prompt_source`. **Verified.**

## Provenance and audit procedure

`scripts/audit_evaluation_results.py` reads these retained raw result payloads, checks that committed rounded final-val values are derived from raw candidate scores, and prints a machine-readable summary. It does not generate images, invoke a model, or change frozen artifacts.
