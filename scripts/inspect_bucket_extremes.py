"""Check whether extreme-aspect-ratio bucket crops (trainer/prepare_data.py's
resize_center_crop) truncate the pose skeleton -- i.e. whether hands, feet, or
other extremities near the long edges of a very tall/wide source image get cut
off when resize_center_crop() scales+crops to the nearest REFERENCE_KREA_BUCKETS
target. Runs against the full dataset stats, CPU-only, no VAE needed.

Two-tier like the rest of this project's smoke tests: a cheap per-image clipping
metric computed for every image assigned to the target bucket(s) (CSV output for
all of them), then a red/green overlay visualization rendered only for the worst
N cases per bucket, for manual review.

Usage:
    python scripts/inspect_bucket_extremes.py \
        --data-root data/full --stats-root data/stats \
        --out-dir data/review/bucket_extremes \
        --buckets 704x1472,1472x704 --worst-n 24
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# must match trainer/prepare_data.py's BUCKETS exactly
REFERENCE_KREA_BUCKETS = [
    (1024, 1024),
    (896, 1152), (1152, 896),
    (832, 1216), (1216, 832),
    (768, 1344), (1344, 768),
    (704, 1472), (1472, 704),
]


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    return min(REFERENCE_KREA_BUCKETS, key=lambda b: abs(np.log(b[0] / b[1]) - np.log(ar)))


def nonzero_bbox(img: Image.Image, thresh: int = 10):
    """Bounding box (left, top, right, bottom) of pixels above threshold, in the
    image's own original pixel coords. right/bottom are exclusive. None if empty."""
    g = np.asarray(img.convert("L"))
    ys, xs = np.where(g > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_geometry(w: int, h: int, tw: int, th: int):
    """Mirrors resize_center_crop's math exactly, without touching pixels --
    returns (scale, resized_w, resized_h, crop_left, crop_top)."""
    scale = max(tw / w, th / h)
    rw, rh = round(w * scale), round(h * scale)
    left, top = (rw - tw) // 2, (rh - th) // 2
    return scale, rw, rh, left, top


def clip_amounts(bbox, scale, left, top, tw, th):
    """bbox is in ORIGINAL pixel coords. Scale to resized coords, then measure
    how far it extends past each edge of the crop window (0 if fully inside)."""
    bx0, by0, bx1, by1 = [c * scale for c in bbox]
    crop_x0, crop_y0, crop_x1, crop_y1 = left, top, left + tw, top + th
    clip_left = max(0.0, crop_x0 - bx0)
    clip_top = max(0.0, crop_y0 - by0)
    clip_right = max(0.0, bx1 - crop_x1)
    clip_bottom = max(0.0, by1 - crop_y1)
    return clip_left, clip_top, clip_right, clip_bottom


def parse_buckets(spec: str):
    out = []
    for tok in spec.split(","):
        tw, th = tok.lower().split("x")
        out.append((int(tw), int(th)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/full")
    ap.add_argument("--stats-root", default="data/stats")
    ap.add_argument("--out-dir", default="data/review/bucket_extremes")
    ap.add_argument("--buckets", default="704x1472,1472x704",
                     help="comma-separated twxth target buckets to inspect")
    ap.add_argument("--worst-n", type=int, default=24,
                     help="render overlay images for the N worst (most-clipped) "
                          "cases per bucket")
    ap.add_argument("--thresh", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_buckets = set(parse_buckets(args.buckets))
    for tb in target_buckets:
        assert tb in REFERENCE_KREA_BUCKETS, f"{tb} is not in REFERENCE_KREA_BUCKETS"

    ar_full = pd.read_csv(os.path.join(args.stats_root, "aspect_ratios.csv"))
    ar = ar_full.copy()
    ar["stem"] = ar["filename"].apply(lambda f: os.path.splitext(f)[0])
    ar["bucket"] = [pick_bucket(w, h) for w, h in zip(ar["width"], ar["height"])]
    ar = ar[ar["bucket"].isin(target_buckets)].copy()
    print(f"{len(ar)} images assigned to target bucket(s) {sorted(target_buckets)} "
          f"out of {len(ar_full)} total")

    controls_dir = os.path.join(args.data_root, "conditioning_images")
    rows = []
    for _, r in ar.iterrows():
        stem = r["stem"]
        ctrl_path = os.path.join(controls_dir, stem + ".png")
        if not os.path.exists(ctrl_path):
            rows.append({"stem": stem, "bucket": "", "max_clip_px": np.nan,
                         "clip_frac": np.nan, "note": "", "error": "control image not found"})
            continue

        img = Image.open(ctrl_path).convert("RGB")
        w, h = img.size
        note = ""
        if (w, h) != (int(r["width"]), int(r["height"])):
            note = f"size mismatch: csv=({r['width']}x{r['height']}) actual=({w}x{h})"

        tw, th = r["bucket"]
        bbox = nonzero_bbox(img, args.thresh)
        if bbox is None:
            rows.append({"stem": stem, "bucket": f"{tw}x{th}", "max_clip_px": np.nan,
                         "clip_frac": np.nan, "note": note, "error": "empty control (no nonzero pixels)"})
            continue

        scale, rw, rh, left, top = crop_geometry(w, h, tw, th)
        cl, ct, cr, cb = clip_amounts(bbox, scale, left, top, tw, th)
        max_clip_px = max(cl, ct, cr, cb)
        # normalize by the target bucket's shorter side so severity is
        # comparable across the differently-shaped buckets
        clip_frac = max_clip_px / min(tw, th)

        rows.append({
            "stem": stem, "bucket": f"{tw}x{th}", "orig_w": w, "orig_h": h,
            "clip_left_px": round(cl, 1), "clip_top_px": round(ct, 1),
            "clip_right_px": round(cr, 1), "clip_bottom_px": round(cb, 1),
            "max_clip_px": round(max_clip_px, 1), "clip_frac": round(clip_frac, 4),
            "note": note, "error": "",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "bucket_extreme_clipping.csv")
    df.to_csv(csv_path, index=False)

    ok = df[df["error"] == ""]
    n_clipped = int((ok["max_clip_px"] > 0).sum()) if len(ok) else 0
    print(f"\n{len(ok)} images checked, {n_clipped} have >0px of skeleton bbox "
          f"outside the crop window")
    if len(ok):
        print(f"clip_frac (clipped px / target short side) stats:\n{ok['clip_frac'].describe()}")

    mismatches = df[df["note"] != ""]
    if len(mismatches):
        print(f"\n{len(mismatches)} control images have a size mismatch vs "
              f"aspect_ratios.csv -- see 'note' column in {csv_path}")
    errored = df[df["error"] != ""]
    if len(errored):
        print(f"{len(errored)} rows errored (missing/empty control) -- see "
              f"'error' column in {csv_path}")

    # render worst-N per bucket for manual review
    for bucket_key, g in ok.groupby("bucket"):
        worst = g.sort_values("max_clip_px", ascending=False).head(args.worst_n)
        for _, r in worst.iterrows():
            stem = r["stem"]
            tw, th = [int(x) for x in bucket_key.split("x")]
            ctrl_path = os.path.join(controls_dir, stem + ".png")
            img = Image.open(ctrl_path).convert("RGB")
            w, h = img.size
            scale, rw, rh, left, top = crop_geometry(w, h, tw, th)
            img_r = img.resize((rw, rh), Image.LANCZOS)

            draw_img = img_r.convert("RGB").copy()
            draw = ImageDraw.Draw(draw_img)
            draw.rectangle([left, top, left + tw, top + th], outline=(255, 0, 0), width=3)  # crop window
            bbox = nonzero_bbox(img, args.thresh)
            if bbox is not None:
                bx0, by0, bx1, by1 = [c * scale for c in bbox]
                draw.rectangle([bx0, by0, bx1, by1], outline=(0, 255, 0), width=3)  # skeleton bbox

            out_path = os.path.join(args.out_dir, f"{bucket_key}__{stem}.png")
            draw_img.save(out_path)

        print(f"bucket {bucket_key}: rendered {len(worst)} worst-case overlays "
              f"to {args.out_dir}/{bucket_key}__*.png (red = crop window, "
              f"green = skeleton bbox)")

    print(
        "\nOpen the worst-N overlays: if green extends outside red on any "
        "side, that's exactly what resize_center_crop() would discard. Check "
        "whether it's hands/feet at frame edges (real problem) vs. background "
        "noise in the control render (harmless). If real: consider re-rendering "
        "controls with the subject better centered/padded before finalizing "
        "prepare_data.py, or accept the loss if it's rare/minor."
    )


if __name__ == "__main__":
    main()