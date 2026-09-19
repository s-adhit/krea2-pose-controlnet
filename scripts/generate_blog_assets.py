#!/usr/bin/env python3
"""Generate publication figures and tables from frozen blog evidence only.

This script intentionally reads no model weights, checkpoints, or mutable result
directories. Exact benchmark values below are transcribed verbatim from the
audited frozen result record named in EVALUATION_SOURCES.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "blog-assets"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"

EVALUATION_SOURCES = (
    "docs/blog-evidence/EVALUATION_RESULTS.md; "
    "docs/blog-evidence/EVALUATION_METHODS.md; "
    "docs/blog-evidence/evaluation_methods.json"
)
TRAINING_SOURCE = "docs/blog-evidence/TRAINING_METHODS_INFRA.md; docs/blog-evidence/training_methods_infra.json"
RELEASE_SOURCE = "docs/evaluation/release/final_release_v1.json"

# Exact values from the frozen result record. Chart labels round only at render.
FINAL_VAL = [
    ("Turbo baseline", 48, 48, 95, 0.0356217617, 0.1101036269, 0.3387305699, 0.3440154150),
    ("parent-4000", 48, 48, 94, 0.4300518135, 0.5725388601, 0.7104922280, 0.3364916809),
    ("mix-025", 48, 48, 95, 0.4520725389, 0.6042746114, 0.7240932642, 0.3369378586),
    ("mix-050", 48, 48, 95, 0.4443005181, 0.5990932642, 0.7130829016, 0.3341130456),
    ("mix-075", 48, 48, 94, 0.4475388601, 0.6036269430, 0.7085492228, 0.3354207429),
    ("A4300", 48, 48, 93, 0.4404145078, 0.5939119171, 0.7169689119, 0.3369788169),
]
NATIVE_DYNAMIC = [
    ("Native cached\ngeometry", 5, 5, 14, 14, 0.2920353982, 0.4026548673, 0.5884955752, 0.3074624562),
    ("Dynamic-768\nbucket", 5, 5, 14, 14, 0.2477876106, 0.3584070796, 0.5309734513, 0.3096808381),
]
HARD_POSE = [
    ("Hard single-person", 8, 8, 8, 0.3727272727, 0.5181818182, 0.6818181818, 0.3281741025),
    ("Hard multi-person", 4, 4, 12, 0.2916666667, 0.4097222222, 0.5625000000, 0.3174141528),
]
PROMPT_INJECTION = [
    ("Normal mix-025\nfinal-val", 48, 48, 95, 0.4520725389, 0.6042746114, 0.7240932642, 0.3369378586),
    ("mix-025 injected\nprompt", 48, 48, 96, 0.4009067358, 0.5563471503, 0.6832901554, 0.3394259131),
]

# Purple-obsidian theme shared by every raster/vector output.
BG = "#101017"
PANEL = "#191725"
PANEL_2 = "#211d31"
TEXT = "#F3EEFF"
MUTED = "#B9B0CC"
GRID = "#5D5572"
VIOLET = "#9D7BFF"
LAVENDER = "#D9CCFF"
PURPLE = "#6748B8"
MAGENTA = "#D486FF"
TEAL = "#74D9C8"
BASELINE = "#8E8A9C"
COLORS = [VIOLET, LAVENDER, MAGENTA]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT, "xtick.color": TEXT, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.alpha": 0.45,
        "axes.grid": True, "grid.linewidth": 0.7, "axes.axisbelow": True,
    })


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def title(ax: plt.Axes, heading: str, subheading: str | None = None) -> None:
    ax.set_title(heading, loc="left", fontsize=16, fontweight="bold", pad=16)
    if subheading:
        ax.text(0, 1.015, subheading, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], label: str,
        accent: str = VIOLET, fontsize: float = 10) -> FancyBboxPatch:
    patch = FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.016,rounding_size=0.02",
                           linewidth=1.4, edgecolor=accent, facecolor=PANEL_2)
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT, wrap=True)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = LAVENDER) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color))


def objective_overview() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(BG)
    ax.text(0.02, 0.95, "Training objective", fontsize=21, fontweight="bold", color=TEXT)
    ax.text(0.02, 0.905, "Frozen production objective: flow-matching MSE + pose-consistency Huber", fontsize=10, color=MUTED)
    box(ax, (0.04, 0.66), (0.20, 0.14), "Clean latent $x_0$\nGaussian noise $\\epsilon$", VIOLET)
    box(ax, (0.31, 0.66), (0.25, 0.14), "$x_t = t\\epsilon + (1-t)x_0$\nnoisy image latent", LAVENDER, 12)
    box(ax, (0.65, 0.66), (0.25, 0.14), "Pose-conditioned velocity\nprediction", MAGENTA)
    arrow(ax, (0.24, 0.73), (0.31, 0.73))
    arrow(ax, (0.56, 0.73), (0.65, 0.73))
    box(ax, (0.31, 0.42), (0.25, 0.12), "Target: $v = \\epsilon - x_0$", TEAL, 12)
    box(ax, (0.65, 0.42), (0.25, 0.12), "Flow MSE\n$\\mathrm{MSE}(\\hat v, v)$", VIOLET, 12)
    arrow(ax, (0.44, 0.66), (0.44, 0.54), TEAL)
    arrow(ax, (0.78, 0.66), (0.78, 0.54), VIOLET)
    arrow(ax, (0.56, 0.48), (0.65, 0.48), VIOLET)
    box(ax, (0.05, 0.19), (0.20, 0.15), "$\\hat x_0 = x_t - t\\hat v$\nVAE decode to unit RGB", LAVENDER, 10)
    box(ax, (0.33, 0.19), (0.20, 0.15), "Frozen fixed-box\nKeypoint R-CNN\n(COCO_V1)", MAGENTA, 10)
    box(ax, (0.61, 0.19), (0.20, 0.15), "Normalized-coordinate\nHuber pose loss", VIOLET, 10)
    box(ax, (0.83, 0.19), (0.13, 0.15), "Total\n$L=L_{flow}+0.04L_{pose}$", TEAL, 9)
    arrow(ax, (0.78, 0.42), (0.15, 0.34), LAVENDER)
    arrow(ax, (0.25, 0.265), (0.33, 0.265), MAGENTA)
    arrow(ax, (0.53, 0.265), (0.61, 0.265), VIOLET)
    arrow(ax, (0.81, 0.265), (0.83, 0.265), TEAL)
    arrow(ax, (0.78, 0.42), (0.895, 0.34), TEAL)
    ax.text(0.05, 0.08, "Pose term is evaluated only when the final shifted timestep is in [0.10, 0.20].", color=MUTED, fontsize=10)
    ax.text(0.05, 0.04, "Image latent is noised; rendered-skeleton control latent remains clean and spatially aligned.", color=MUTED, fontsize=10)
    finish(fig, "training_objective_overview")


def training_lineage() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.8))
    ax.set_xlim(-100, 5300); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("Optimizer step")
    title(ax, "Endpoint training lineage", "Measured optimizer-loop time; excludes checkpoint serialization, mirroring, startup, and inter-run gaps.")
    # Solid segments are endpoint-relevant; faded dashed extensions retain the
    # complete run extent without making their later runtime part of the total.
    ax.plot([0, 3000], [0.67, 0.67], color=VIOLET, linewidth=18, solid_capstyle="round")
    ax.plot([3000, 4000], [0.67, 0.67], color=PURPLE, linewidth=18, solid_capstyle="butt")
    ax.plot([4000, 5000], [0.67, 0.67], color=PURPLE, linewidth=12, linestyle="--", alpha=0.36)
    ax.plot([4000, 4300], [0.36, 0.36], color=MAGENTA, linewidth=18, solid_capstyle="round")
    ax.plot([4300, 4500], [0.36, 0.36], color=MAGENTA, linewidth=12, linestyle="--", alpha=0.36)
    ax.text(1500, 0.79, "Production 0–3000\n12:23:40", ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
    ax.text(3500, 0.79, "Cooldown 3000–5000\nselected slice: 3001–4000 = 4:10:16", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")
    ax.text(3800, 0.47, "Finish-control branch 4000–4500\nselected slice: 4001–4300 = 1:17:09", ha="center", va="bottom", fontsize=9.5, color=TEXT, fontweight="bold")
    for step, y in [(4000, 0.67), (4300, 0.36)]:
        ax.plot(step, y, "o", color=TEAL, markersize=10, zorder=4)
    ax.annotate("parent-4000 selected", (4000, 0.67), xytext=(3740, 0.93), color=TEXT, fontsize=9, ha="center",
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.2})
    ax.annotate("A4300 selected", (4300, 0.36), xytext=(4560, 0.18), color=TEXT, fontsize=9, ha="center",
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.2})
    ax.annotate("mix-025 final release\n0.75 parent-4000 + 0.25 A4300", (4300, 0.36), xytext=(4800, 0.78),
                color=TEXT, fontsize=10, fontweight="bold", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": LAVENDER, "lw": 1.5})
    ax.text(0.5, 0.07, "Endpoint-relevant optimizer-loop total: 17:51:05", transform=ax.transAxes,
            ha="center", color=LAVENDER, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.01, "Cooldown and finish-control lines show full run extent; labels mark the endpoint-selected slices.", transform=ax.transAxes,
            ha="center", color=MUTED, fontsize=9)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 4300, 4500, 5000])
    finish(fig, "training_lineage")


def grouped_pck(name: str, rows: list[tuple], heading: str, subtitle: str, labels: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    labels = labels or [r[0] for r in rows]
    values = np.array([[r[-4], r[-3], r[-2]] for r in rows])
    x = np.arange(len(rows)); width = 0.23
    for i, (metric, color) in enumerate(zip(["PCK@.05", "PCK@.10", "PCK@.20"], COLORS)):
        bars = ax.bar(x + (i - 1) * width, values[:, i], width, label=metric, color=color, edgecolor=BG, linewidth=0.8)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values[:, i]], padding=3, fontsize=8, color=TEXT, rotation=90)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Globally pooled PCK")
    ax.set_xticks(x, labels); ax.legend(ncol=3, frameon=False, labelcolor=TEXT, loc="upper left")
    title(ax, heading, subtitle)
    finish(fig, name)


def final_val_pck() -> None:
    selected = [FINAL_VAL[i] for i in [0, 1, 2, 5]]
    grouped_pck("final_val_pck", selected, "Final-val pose adherence", "48 attempted/evaluated images; native geometry; values shown rounded from frozen payloads.")


def final_val_clip() -> None:
    selected = [FINAL_VAL[i] for i in [0, 1, 2, 5]]
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    x = np.arange(len(selected)); values = [r[-1] for r in selected]
    bars = ax.bar(x, values, color=[BASELINE, VIOLET, LAVENDER, MAGENTA], width=0.58, edgecolor=BG)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, color=TEXT, fontsize=10)
    ax.set_ylim(0, 0.40); ax.set_ylabel("Mean image/prompt CLIP cosine similarity")
    ax.set_xticks(x, [r[0] for r in selected]); title(ax, "Final-val prompt/image similarity", "Absolute 0–0.40 axis retained because candidate differences are small; no significance claim.")
    finish(fig, "final_val_clip")


def paired_metrics(name: str, rows: list[tuple], heading: str, subtitle: str, note: str) -> None:
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1.65, 0.75]})
    fig.subplots_adjust(wspace=0.22)
    labels = [r[0] for r in rows]; values = np.array([[r[-4], r[-3], r[-2]] for r in rows])
    x = np.arange(len(rows)); width = 0.22
    for i, (metric, color) in enumerate(zip(["PCK@.05", "PCK@.10", "PCK@.20"], COLORS)):
        bars = ax.bar(x + (i - 1) * width, values[:, i], width, label=metric, color=color, edgecolor=BG)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values[:, i]], padding=3, fontsize=8, color=TEXT, rotation=90)
    ax.set_ylim(0, 1); ax.set_ylabel("Globally pooled PCK"); ax.set_xticks(x, labels); ax.legend(ncol=3, frameon=False, labelcolor=TEXT, loc="upper left")
    clip = [r[-1] for r in rows]
    bars = cx.bar(np.arange(len(rows)), clip, color=[VIOLET, MAGENTA], edgecolor=BG, width=0.58)
    cx.bar_label(bars, labels=[f"{v:.4f}" for v in clip], padding=4, fontsize=9, color=TEXT)
    cx.set_ylim(0, 0.40); cx.set_ylabel("Mean CLIP cosine")
    cx.set_xticks(np.arange(len(rows)), labels, fontsize=9)
    fig.suptitle(heading, x=0.065, y=0.98, ha="left", fontsize=16, color=TEXT, fontweight="bold")
    fig.text(0.065, 0.925, subtitle, color=MUTED, fontsize=9)
    fig.text(0.065, 0.02, note, color=MUTED, fontsize=9)
    finish(fig, name)


def comparison_figures() -> None:
    paired_metrics("native_vs_dynamic", NATIVE_DYNAMIC, "Native cached geometry vs dynamic-768", "Five attempted/evaluated conditions per mode; mix-025, Turbo 8 steps, CFG 0, μ=1.15, control scale 1.0.", "Geometry/control-input-path ablation, not a pure resolution-only experiment: bucket, crop framing, control raster, and VAE control encoding differ.")
    paired_metrics("hard_pose_single_multi", HARD_POSE, "Hard-pose stress: single-person vs multi-person", "12 fixed-seed generations total; mix-025, native geometry, control scale 1.0.", "Subsets differ in image and person count (8 single-person images / 4 multi-person images); descriptive frozen stress results only.")
    paired_metrics("prompt_injection_effect", PROMPT_INJECTION, "Prompt injection effect", "48 final-val stems; mix-025, native geometry, Turbo 8 steps, CFG 0, μ=1.15, control scale 1.0.", "Same frozen final-val stems; prompt text changed; control and evaluation setup otherwise fixed. CLIP uses each condition's actual evaluation prompt.")


def infra_summary(training: dict) -> None:
    infra, recipe = training["infrastructure"], training["recipe"]
    fig, ax = plt.subplots(figsize=(14.5, 8.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.94, "Training infrastructure summary", fontsize=20, fontweight="bold", color=TEXT)
    ax.text(0.03, 0.90, "Frozen normal-host and final-lineage runtime record", fontsize=10, color=MUTED)
    groups = [
        ("Host", ["NVIDIA GH200 480GB", "94.5 GiB PyTorch-visible", "ARM64 / aarch64", "Ubuntu 22.04.5"], 0.05, 0.55),
        ("Software", ["PyTorch 2.7.0", "CUDA 12.8", "cuDNN 9.8", "single GPU / world size 1"], 0.37, 0.55),
        ("Runtime", ["BF16 autocast", "no torch.compile", "no fused AdamW", "no gradient checkpointing"], 0.69, 0.55),
        ("Batching & cache", ["microbatch 1", "gradient accumulation 32", "effective batch 32", "cached text + VAE latents on NFS"], 0.21, 0.15),
    ]
    for heading, items, x, y in groups:
        w = 0.27 if y > 0.3 else 0.58
        h = 0.27
        box(ax, (x, y), (w, h), "", VIOLET)
        ax.text(x + 0.025, y + h - 0.055, heading, color=LAVENDER, fontsize=12, fontweight="bold", ha="left")
        for i, item in enumerate(items):
            ax.text(x + 0.03, y + h - 0.105 - 0.04 * i, "• " + item, color=TEXT, fontsize=10, ha="left")
    ax.text(0.03, 0.05, "“480GB” is the PyTorch-reported device/product name, not an HBM-capacity claim.", color=MUTED, fontsize=9)
    finish(fig, "infra_summary")


def table_file(name: str, body: str) -> None:
    (TABLES / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def make_tables(training: dict, methods: dict, release: dict) -> None:
    state = training["recipe"]["trainable_state"]
    recipe = training["recipe"]
    bucket_string = ", ".join(f"{w}×{h}" for w, h in recipe["cache"]["buckets"])
    table_file("training_recipe.md", f"""# Frozen training recipe

Source: `{TRAINING_SOURCE}`.

| Setting | Frozen value |
| --- | --- |
| Base model | Krea-2 Raw |
| LoRA rank | {state['lora_rank']} |
| Trainable parameters | {state['parameters']:,} float32 |
| Trainable tensor count | {state['tensor_count']} |
| ControlInputLayer parameters | {state['control_input_parameters']:,} |
| LoRA parameters | {state['lora_parameters']:,} |
| Optimizer | AdamW |
| Betas | (0.9, 0.99) |
| Epsilon | 1e-8 |
| Weight decay | 0.0 |
| Max grad norm | 1.0 |
| Microbatch | 1 |
| Gradient accumulation | 32 |
| Effective batch | 32 |
| Caption dropout | 0.10 |
| Control dropout | 0.0 |
| Precision | CUDA BF16 autocast |
| Workers / prefetch | 4 / 4 (persistent workers; pin memory) |
| Bucket scheme | {bucket_string} |
| Gradient checkpointing | Disabled (0 blocks) |
| torch.compile | Disabled |
| Fused AdamW | Disabled |
""")
    table_file("training_stages.md", """# Final-lineage training stages

Source: `docs/blog-evidence/TRAINING_METHODS_INFRA.md` and `docs/blog-evidence/RUNTIME_AUDIT.md`. Runtime is measured optimizer-loop time, not complete wall-clock time.

| Stage | Step range | LR schedule | Pose-loss setting | Selected checkpoint | Optimizer-loop runtime | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| Production | 0–3000 | 200-step warmup to 1e-4, then 1e-4 | Production objective; λ_pose 0.04 when active | Feeds cooldown | 12:23:40 | Initial production |
| Cooldown | 3000–5000 run; parent selected at 4000 | cosine 1e-4 → 1e-5 over 2000 updates | Production objective; λ_pose 0.04 when active | parent-4000 | 3001–4000: 4:10:16 | Continuation / parent selection |
| Finish-control branch | 4000–4500 run; A4300 selected | cosine 2e-5 → 5e-6 over 500 updates | λ_pose = 0.04 constant | finish-control-a4300 | 4001–4300: 1:17:09 | Finish-control selection |
| mix-025 release | n/a | n/a | n/a | final release | No training runtime | 0.75 parent-4000 + 0.25 A4300 trainable-tensor interpolation |
| Endpoint-relevant total | 0–3000 + 3001–4000 + 4001–4300 | — | — | mix-025 inputs | **17:51:05** | Selected endpoint lineage only |
""")
    table_file("evaluation_method_summary.md", f"""# Evaluation method summary

Source: `{EVALUATION_SOURCES}`.

| Metric / process | Exact definition | Threshold / model | Aggregation | Failure handling | Important caveat |
| --- | --- | --- | --- | --- | --- |
| PCK | For each eligible reference joint, 1[d ≤ threshold × reference-person valid-joint extent diagonal]; inclusive comparison in generated-image coordinates | Thresholds .05, .10, .20 | Global correct eligible joints / global eligible joints | Missing valid detector joint and unmatched eligible reference remain denominator zeros | Not per-image/person mean; normalization is extent diagonal, not bbox/torso/head scale |
| Person matching | One-to-one Hungarian assignment using mean unnormalized pixel Euclidean distance over shared valid joints | Keypoint R-CNN ResNet50-FPN COCO_V1; person and keypoint confidence ≥ .5 | Finite-cost assigned reference/generated pairs; `matched` is a person-pair count | No-shared-joint pairs forbidden; unmatched references score zero eligible joints | No match-distance cutoff; extra generated people are reported but not directly PCK-penalized |
| CLIP | Explicit cosine of generated-image and exact evaluation-prompt embeddings | `openai/clip-vit-base-patch32` | Arithmetic mean over every completed generated image | Missing required generation raises; no silent drop | Independent of PCK matching; prompt injection uses injected prompt text |
| Native geometry | Persisted paired cached-latent bucket with validated recorded resize/crop geometry; cached control latent consumed at generation | Native aspect-preserving cached geometry | PCK source references transformed through native geometry | Frozen scoring requires complete generation set | Original paired RGB/control preprocessing used shared geometry |
| Dynamic-768 geometry | Closest log-aspect bucket, resize-to-cover, center crop, authoritative control VAE encode with fixed sampling seed; source RGB not used | Nine frozen buckets, including 768×768 and aspect buckets | PCK source references transformed through dynamic geometry | Frozen five-condition ablation requires complete set | Not a pure resolution-only experiment: bucket, crop framing, control raster, and control encoding differ |
""")
    final_rows = "\n".join(f"| {n} | {a} / {e} | {m} | {p05:.10f} | {p10:.10f} | {p20:.10f} | {clip:.10f} |" for n,a,e,m,p05,p10,p20,clip in FINAL_VAL)
    table_file("final_val_results.md", f"""# Frozen final-val results

Source: `docs/blog-evidence/EVALUATION_RESULTS.md`. Native geometry; Krea-2 Turbo, 8 steps, CFG 0, μ=1.15. All values retain frozen payload precision.

| Candidate | Attempted / evaluated | Matched | PCK@.05 | PCK@.10 | PCK@.20 | CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{final_rows}
""")
    table_file("ablation_results.md", """# Frozen ablation and stress results

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
""")
    endpoints = release["candidate"]["interpolation"]["endpoints"]
    public_path = "release/krea2-pose-control-mix025.safetensors"
    nfs_path = "/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors"
    table_file("release_identity.md", f"""# Release identity

Primary source: `{RELEASE_SOURCE}`. Safetensors public filename and frozen NFS materialization path/SHA are recorded in `docs/release/HF_MODEL_CARD.md` and the frozen canonical-generation summary.

| Field | Frozen value |
| --- | --- |
| Release ID | {release['release_id']} |
| Candidate | mix-025 |
| parent-4000 path / SHA | `{endpoints[0]['path']}` / `{endpoints[0]['sha256']}` |
| A4300 path / SHA | `{endpoints[1]['path']}` / `{endpoints[1]['sha256']}` |
| Interpolation formula | 0.75 × parent-4000 + 0.25 × A4300; float32; `state['model']` trainable control/LoRA tensors only |
| Release safetensors filename / public path | `krea2-pose-control-mix025.safetensors` / `{public_path}` |
| Release safetensors materialization path | `{nfs_path}` |
| Release safetensors SHA256 | `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1` |
| Trainable parameters | {state['parameters']:,} |
| Trainable tensor count | {state['tensor_count']} |
| Runtime defaults | Krea-2 Turbo; 8 steps; CFG 0; μ 1.15; control scale 1.0; native cached geometry |
""")


def asset_index() -> None:
    entries = [
        ("Training method", "figures/training_objective_overview.png (+ SVG)", "Flow-matching and gated pose-Huber objective.", TRAINING_SOURCE, "Pose loss is active only for shifted t in [0.10, 0.20].", "Final"),
        ("Training lineage", "figures/training_lineage.png (+ SVG)", "Selected endpoint lineage and endpoint-relevant optimizer-loop time.", TRAINING_SOURCE + "; docs/blog-evidence/RUNTIME_AUDIT.md", "Optimizer-loop time is not complete wall-clock time.", "Final"),
        ("Final validation", "figures/final_val_pck.png (+ SVG)", "PCK comparison of Turbo baseline, parent-4000, mix-025, and A4300.", "docs/blog-evidence/EVALUATION_RESULTS.md", "48 images; globally pooled PCK; rounded labels only.", "Final"),
        ("Final validation", "figures/final_val_clip.png (+ SVG)", "CLIP comparison of the same four candidates.", "docs/blog-evidence/EVALUATION_RESULTS.md", "Absolute 0–0.40 axis; small differences are not exaggerated.", "Final"),
        ("Geometry ablation", "figures/native_vs_dynamic.png (+ SVG)", "Native cached geometry versus dynamic-768.", EVALUATION_SOURCES, "Geometry/control-input-path ablation, not pure resolution-only.", "Final"),
        ("Stress evaluation", "figures/hard_pose_single_multi.png (+ SVG)", "Hard single-person versus hard multi-person results.", "docs/blog-evidence/EVALUATION_RESULTS.md", "Different subset sizes; descriptive stress result.", "Final"),
        ("Prompting", "figures/prompt_injection_effect.png (+ SVG)", "Normal final-val prompts versus injected prompts.", "docs/blog-evidence/EVALUATION_RESULTS.md", "Same 48 stems; prompt text changed; CLIP follows actual prompt.", "Final"),
        ("Methods / infrastructure", "figures/infra_summary.png (+ SVG)", "Final-lineage host, runtime, batching, and cache settings.", TRAINING_SOURCE, "GH200 480GB is a device-reported name, not HBM capacity.", "Final"),
        ("Methods", "tables/training_recipe.md", "Frozen recipe and runtime flags.", TRAINING_SOURCE, "Final-lineage settings only.", "Final"),
        ("Training lineage", "tables/training_stages.md", "Stage schedules, selections, and measured loop timing.", TRAINING_SOURCE, "No unsupported complete wall-clock claim.", "Final"),
        ("Evaluation methods", "tables/evaluation_method_summary.md", "Definitions and limits for metrics and geometry.", EVALUATION_SOURCES, "Detector-/geometry-dependent measures.", "Final"),
        ("Final validation", "tables/final_val_results.md", "Exact frozen final-val candidate values.", "docs/blog-evidence/EVALUATION_RESULTS.md", "Exact payload precision retained.", "Final"),
        ("Ablations", "tables/ablation_results.md", "Exact geometry, stress, and prompt-injection values/counts.", "docs/blog-evidence/EVALUATION_RESULTS.md", "Counts retained where available.", "Final"),
        ("Release", "tables/release_identity.md", "Release candidate, endpoint identity, artifact, and runtime defaults.", RELEASE_SOURCE, "Release identity is frozen; no artifact is modified.", "Final"),
    ]
    rows = "\n".join(f"| {section} | `{path}` | {shows} | `{source}` | {caveat} | {status} |" for section, path, shows, source, caveat, status in entries)
    (OUT / "BLOG_ASSET_INDEX.md").write_text(f"""# Blog asset index

All assets in this directory are generated by `scripts/generate_blog_assets.py` from frozen evidence records. The pre-existing architecture figure is user-approved and intentionally unchanged; it is not recreated or indexed as a new asset here.

| Intended blog section | Path | What it shows | Frozen evidence source | Caption / caveat | Status |
| --- | --- | --- | --- | --- | --- |
{rows}

## Reproduction

```bash
MPLCONFIGDIR=/tmp/krea2-mpl uv run python scripts/generate_blog_assets.py
```

Figures use the purple-obsidian theme and are exported as high-resolution PNG plus SVG. Exact benchmark precision is preserved in the Markdown tables and generator source; chart labels are presentation rounding only.
""", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    training = json.loads((ROOT / "docs/blog-evidence/training_methods_infra.json").read_text())
    methods = json.loads((ROOT / "docs/blog-evidence/evaluation_methods.json").read_text())
    release = json.loads((ROOT / "docs/evaluation/release/final_release_v1.json").read_text())
    setup_style()
    objective_overview(); training_lineage(); final_val_pck(); final_val_clip(); comparison_figures(); infra_summary(training)
    make_tables(training, methods, release); asset_index()
    print(f"generated figures in {FIGURES}")
    print(f"generated tables in {TABLES}")


if __name__ == "__main__":
    main()
