"""Nominate candidates for the 20-40 image manual diagnostic panel.

This does NOT pick your final diagnostic set automatically -- picking that set
is a judgment call that should stay human-driven. What this script does is cut
the search space down from 17,495 images to a small, stratified shortlist per
stratum, rendered as contact sheets, so you can eyeball each stratum and
hand-pick 2-5 examples from it.

Strata used (from your existing audit/bucket-analysis outputs):
    - control sparsity: low / high nonzero fraction
    - aspect ratio: extreme portrait / extreme landscape / normal
    - source (if recoverable from metadata): coco / danbooru / human-art *
    - solo vs crowd: NOT auto-detected here (no person-count column exists
      yet) -- included as a TODO so you remember to either derive it or
      hand-tag it while reviewing the contact sheets.

Inputs expected (adjust paths to match your project):
    data/stats/aspect_ratios.csv        columns: file_name, aspect_ratio, width, height
    data/stats/control_sparsity.csv     columns: file_name, nonzero_fraction
    data/full/metadata.jsonl            {"file_name": ..., "text": ..., "source": ... (optional)}

Output:
    data/review/candidates_<stratum>.jsonl   shortlist per stratum
    data/review/contact_sheet_<stratum>.png  grid of RGB + control thumbnails

Usage:
    python scripts/select_diagnostic_candidates.py \
        --data-root data/full --stats-root data/stats --out-root data/review \
        --per-stratum 12
"""

import argparse
import json
import math
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

THUMB = 220


def load_join(stats_root: str, data_root: str) -> pd.DataFrame:
    ar = pd.read_csv(os.path.join(stats_root, "aspect_ratios.csv"))
    sp = pd.read_csv(os.path.join(stats_root, "control_sparsity.csv"))

    sp = sp[sp["ok"] == True].copy()
    sp["filename"] = sp["path"].apply(lambda p: os.path.basename(str(p)))
    sp = sp.drop(columns=["path", "width", "height", "ok", "error"], errors="ignore")

    ar["stem"] = ar["filename"].apply(lambda f: os.path.splitext(f)[0])
    sp["stem"] = sp["filename"].apply(lambda f: os.path.splitext(f)[0])
    df = ar.merge(sp, on="stem", how="inner", suffixes=("", "_ctrl"))
    df = df.rename(columns={"filename": "file_name"})  # this is the .jpg name -> used for images/
    df["control_file_name"] = df["stem"] + ".png"       # explicit .png name -> used for conditioning_images/

    meta_path = os.path.join(data_root, "metadata.jsonl")
    sources = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "source" in rec:
                    sources[rec["file_name"]] = rec["source"]
    df["source"] = df["file_name"].apply(derive_source)
    return df

def derive_source(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    if stem.startswith("coco_"):
        return "coco"
    if stem.startswith("danbooru"):
        return "danbooru"
    if stem.startswith("painting_humanart"):
        return "humanart_painting"
    if stem.startswith("real_human_humanart"):
        return "humanart_real_human"
    if stem.startswith("sculpture_humanart"):
        return "humanart_sculpture"
    return "unknown"


def build_strata(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strata = {}

    sp_lo, sp_hi = df["nonzero_fraction"].quantile([0.1, 0.9])
    strata["sparsity_low"] = df[df["nonzero_fraction"] <= sp_lo]
    strata["sparsity_high"] = df[df["nonzero_fraction"] >= sp_hi]

    strata["ar_extreme_portrait"] = df[df["aspect_ratio"] < 0.5]
    strata["ar_extreme_landscape"] = df[df["aspect_ratio"] > 2.0]
    strata["ar_normal"] = df[df["aspect_ratio"].between(0.9, 1.1)]

    for src, g in df.groupby("source"):
        if src != "unknown":
            strata[f"source_{src}"] = g

    # TODO: solo vs crowd has no derived column yet. If you run a person
    # detector pass later, add a `person_count` column upstream and split
    # here on person_count == 1 vs > 1. Left out for now rather than faked.

    return strata


def sample_stratum(g: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(g) <= n:
        return g
    return g.sample(n=n, random_state=seed)


def contact_sheet(rows: pd.DataFrame, images_dir: str, controls_dir: str, out_path: str):
    n = len(rows)
    if n == 0:
        return
    cols = min(6, n)
    grid_rows = math.ceil(n / cols)
    cell_w, cell_h = THUMB * 2 + 10, THUMB + 40  # rgb | control side by side, + caption strip
    sheet = Image.new("RGB", (cell_w * cols, cell_h * grid_rows), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (_, row) in enumerate(rows.iterrows()):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        fname = row["file_name"]
        try:
            img = Image.open(os.path.join(images_dir, fname)).convert("RGB")
            img.thumbnail((THUMB, THUMB))
            sheet.paste(img, (x0, y0))
        except Exception as e:
            draw.text((x0, y0), f"img err: {e}", fill="red", font=font)
        try:
            ctrl = Image.open(os.path.join(controls_dir, fname)).convert("RGB")
            ctrl.thumbnail((THUMB, THUMB))
            sheet.paste(ctrl, (x0 + THUMB + 10, y0))
        except Exception as e:
            draw.text((x0 + THUMB + 10, y0), f"ctrl err: {e}", fill="red", font=font)
        draw.text((x0, y0 + THUMB + 2), f"{fname}", fill="black", font=font)
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/full")
    ap.add_argument("--stats-root", default="data/stats")
    ap.add_argument("--out-root", default="data/review")
    ap.add_argument("--per-stratum", type=int, default=12,
                    help="candidates rendered per stratum for you to pick 2-5 from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    images_dir = os.path.join(args.data_root, "images")
    controls_dir = os.path.join(args.data_root, "conditioning_images")

    df = load_join(args.stats_root, args.data_root)
    strata = build_strata(df)

    print(f"{len(strata)} strata built from {len(df)} rows")
    for name, g in strata.items():
        sample = sample_stratum(g, args.per_stratum, args.seed)
        jl_path = os.path.join(args.out_root, f"candidates_{name}.jsonl")
        with open(jl_path, "w") as f:
            for _, row in sample.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        png_path = os.path.join(args.out_root, f"contact_sheet_{name}.png")
        contact_sheet(sample, images_dir, controls_dir, png_path)
        print(f"  {name}: {len(g)} in pool -> {len(sample)} shortlisted -> {png_path}")

    print(
        "\nNext: open each contact_sheet_*.png, hand-pick 2-5 filenames per stratum "
        "(aim for 20-40 total), and list the chosen file_names (one per line) into "
        "data/review/diagnostic_picks.txt for build_val_manifests.py."
    )


if __name__ == "__main__":
    main()