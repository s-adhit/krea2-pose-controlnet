# Critic-alignment diagnostic

Internal metrics are lower-is-better; external PCK is higher-is-better.
Correlations are descriptive only (five checkpoint observations), not statistically conclusive.

| checkpoint | critic loss mean | normalized coordinate error mean | PCK@.05 | PCK@.10 | PCK@.20 |
|---|---:|---:|---:|---:|---:|
| LR-only 1500 @ 5e-5 (1500) | 0.0219920 | 0.2451323 | 0.054612 | 0.180825 | 0.412621 |
| Coordinate Huber 1e-5 10pct t010-020 1525 (1525) | 0.0203166 | 0.2389282 | 0.047330 | 0.152913 | 0.398058 |
| Coordinate Huber 1e-5 10pct t010-020 1550 (1550) | 0.0212522 | 0.2432506 | 0.048544 | 0.162621 | 0.408981 |
| Coordinate Huber 1e-5 10pct t010-020 1575 (1575) | 0.0226737 | 0.2502943 | 0.032767 | 0.144417 | 0.393204 |
| Coordinate Huber 1e-5 10pct t010-020 1600 (1600) | 0.0218753 | 0.2456850 | 0.053398 | 0.144417 | 0.368932 |

## Descriptive correlations

- critic_loss vs pck_005: Pearson=-0.3731969886816621; Spearman=0.0.
- critic_loss vs pck_010: Pearson=-0.08022541969099535; Spearman=-0.20519567041703082.
- critic_loss vs pck_020: Pearson=-0.181983488583704; Spearman=-0.1.
- normalized_coordinate_error vs pck_005: Pearson=-0.5164622996306654; Spearman=-0.1.
- normalized_coordinate_error vs pck_010: Pearson=-0.2267174823327762; Spearman=-0.5642880936468347.
- normalized_coordinate_error vs pck_020: Pearson=-0.23144570772363893; Spearman=-0.5.

Interpretation criteria:

- A. Internal critic improves and PCK improves: aligned / promising.
- B. Internal critic improves and PCK worsens: reward misalignment.
- C. Internal critic does not improve: auxiliary optimization/gradient effectiveness problem.

A well-aligned critic generally has negative internal-error vs PCK correlation.
