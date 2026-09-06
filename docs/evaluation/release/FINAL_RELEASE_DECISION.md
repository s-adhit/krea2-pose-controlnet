# Final release decision: Krea-2 Pose Control-LoRA v1

`mix-025` is frozen as the final candidate. It is a float32 interpolation of
trainable control/LoRA tensors in `state['model']` only:

```text
(1 - 0.25) * parent-4000 + 0.25 * finish-control-a4300
```

The endpoint checkpoints and their SHA-256 values are pinned in
[`final_release_v1.json`](final_release_v1.json). The canonical runtime is
Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, without resolution-dependent `mu`.

Native aspect-preserving cached latent buckets are the default. Dynamic-768 is
still a supported alternative, but native scored PCK `.2920/.4027/.5885` versus
`.2478/.3584/.5310` for dynamic (CLIP `.30746` versus `.30968`).

Use control scale `1.0` by default. Although aggregate best PCK occurred at
`1.5`, performance was non-monotonic by pose class; `1.25-1.50` is an optional
stronger-control range rather than the release default.

On frozen final-val, mix-025 reached PCK `.4521/.6043/.7241`, compared with
parent-4000 `.4301/.5725/.7105` and A4300 `.4404/.5939/.7170`. Its CLIP score
was `.33694` (parent `.33649`, A4300 `.33698`). Against unmodified Turbo, its
PCK was `.4521/.6043/.7241` versus `.0356/.1101/.3387` for Turbo base.

Style-LoRA composition is limited to one independently applied adapter at a
time. Pose and Style-LoRA tensors remain separate, there is no permanent merge,
and runtime application is reversible. The frozen defaults are darkbrush `0.75`
(useful `0.50-0.75`), rainywindow `0.50` (`0.25-0.75`), retroanime `0.50`
(`0.25-0.50`), and realism `0.25` (`0.25-0.75`). Multi-Style-LoRA composition
is not part of the canonical v1 release.

Hard-pose PCK was `.3727/.5182/.6818` for single-person cases and
`.2917/.4097/.5625` for multi-person stress. Overlapping interactions remain a
limitation. No systematic Pose-Control-specific hand regression was observed;
the small hand-heavy subset's local A4300 advantage does not outweigh mix-025's
broader frozen benchmark result.

All source experiment paths and available SHA-256 provenance are frozen in the
machine-readable release contract. The contract's own immutable SHA-256 is
`9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b`, pinned
by `tests/test_final_release_decision.py`.
