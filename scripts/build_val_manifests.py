"""Build the three immutable manifests: diagnostic_val, val, train.

Order enforced: diagnostic_val is picked and REMOVED from the pool BEFORE the
val split is computed, so val/train can never contain a diagnostic example.

Val selection criterion (decided this session): stratify by
(source x AR bucket x sparsity tercile), ~5% proportional allocation per
stratum, fixed seed, never resampled after seeing results. Source and bucket
are the axes that change model behavior (different visual domains, different
crop behavior at train time); sparsity tercile targets this dataset's known
risk factor (1.85% mean nonzero control pixels).

Usage:
    python scripts/build_val_manifests.py \
        --data-root data/full --stats-root data/stats \
        --diagnostic-picks data/review/diagnostic_picks.txt \
        --out-root data/manifests --val-fraction 0.05 --seed 42
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# must match trainer/prepare_data.py BUCKETS exactly
REFERENCE_KREA_BUCKETS = [
    (1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832),
    (768, 1344), (1344, 768), (704, 1472), (1472, 704),
]


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    return min(REFERENCE_KREA_BUCKETS, key=lambda b: abs(np.log(b[0] / b[1]) - np.log(ar)))


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


def load_dataset_table(data_root: str, stats_root: str) -> pd.DataFrame:
    ar = pd.read_csv(os.path.join(stats_root, "aspect_ratios.csv"))
    sp = pd.read_csv(os.path.join(stats_root, "control_sparsity.csv"))

    sp = sp[sp["ok"] == True].copy()
    sp["filename"] = sp["path"].apply(lambda p: os.path.basename(str(p)))
    sp = sp.drop(columns=["path", "width", "height", "ok", "error"], errors="ignore")

    ar["stem"] = ar["filename"].apply(lambda f: os.path.splitext(f)[0])
    sp["stem"] = sp["filename"].apply(lambda f: os.path.splitext(f)[0])
    df = ar.merge(sp, on="stem", how="inner", suffixes=("", "_ctrl"))
    df = df.rename(columns={"filename": "file_name"})

    df["source"] = df["file_name"].apply(derive_source)
    df["bucket"] = [pick_bucket(w, h) for w, h in zip(df["width"], df["height"])]
    df["sparsity_tercile"] = pd.qcut(df["nonzero_fraction"], 3, labels=["low", "mid", "high"])
    return df


def load_diagnostic_picks(path: str) -> set[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Review the contact sheets from "
            "select_diagnostic_candidates.py and write one file_name per line "
            "into this file first (20-40 total)."
        )
    with open(path, encoding="utf-8") as f:
        picks = {line.strip() for line in f if line.strip()}
    return picks


def stratified_val_split(df: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    strat_key = df["source"].astype(str) + "|" + df["bucket"].astype(str) + "|" + df["sparsity_tercile"].astype(str)
    df = df.assign(_strat=strat_key)
    val_files = []
    rng = np.random.RandomState(seed)
    for _, g in df.groupby("_strat"):
        n = min(max(1, round(len(g) * fraction)), len(g))
        idx = rng.choice(g.index, size=n, replace=False)
        val_files.extend(df.loc[idx, "file_name"].tolist())
    return set(val_files)


def write_manifest(path: str, file_names: list[str], data_root: str):
    # metadata.jsonl's file_name field is prefixed "images/<stem>.jpg" for
    # every row (confirmed, not a mixed-format edge case), while
    # aspect_ratios.csv / diagnostic_picks.txt use bare "<stem>.jpg".
    # Normalize both sides to basename so lookups work regardless of prefix.
    captions = {}
    with open(os.path.join(data_root, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = os.path.basename(rec["file_name"])
            if key in captions:
                raise ValueError(
                    f"duplicate metadata.jsonl row for {key!r} -- "
                    f"one has file_name={captions[key]['file_name']!r}, "
                    f"other has file_name={rec['file_name']!r}. Dedup "
                    f"metadata.jsonl before continuing."
                )
            captions[key] = rec

    missing = [fn for fn in file_names if os.path.basename(fn) not in captions]
    if missing:
        raise KeyError(
            f"{len(missing)} file_name(s) in this split have no metadata.jsonl "
            f"row (writing {os.path.basename(path)}):\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nIf this is diagnostic_val.jsonl, these came from "
              "diagnostic_picks.txt -- check for typos against the contact "
              "sheets, or confirm the file is actually missing from "
              "metadata.jsonl entirely."
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for fname in sorted(file_names):
            rec = dict(captions[os.path.basename(fname)])
            # normalize the written-out record too, so every manifest row
            # has a consistent bare-filename file_name regardless of which
            # format the source row used
            rec["file_name"] = os.path.basename(fname)
            f.write(json.dumps(rec) + "\n")
def load_excluded_stems(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["stem"].astype(str))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/full")
    ap.add_argument("--stats-root", default="data/stats")
    ap.add_argument("--diagnostic-picks", default="data/review/diagnostic_picks.txt")
    ap.add_argument("--out-root", default="data/manifests")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--excluded-stems", default="data/review/excluded_stems.csv",
                     help="CSV with a 'stem' column; these images are dropped "
                          "from all splits before val/train split is computed")
    args = ap.parse_args()

    df = load_dataset_table(args.data_root, args.stats_root)
    print(f"{len(df)} total examples")

    diagnostic = load_diagnostic_picks(args.diagnostic_picks)

    excluded = load_excluded_stems(args.excluded_stems)
    if excluded:
        excluded_files = {s + ".jpg" for s in excluded}  # file_name is bare "<stem>.jpg"
        bad_picks = diagnostic & excluded_files
        if bad_picks:
            raise ValueError(
                f"{len(bad_picks)} diagnostic_picks.txt entries are in the "
                f"bucket-crop exclusion list and need manual replacement:\n  "
                + "\n  ".join(sorted(bad_picks))
            )
        before = len(df)
        df = df[~df["file_name"].isin(excluded_files)]
        print(f"excluded {before - len(df)} images (clip_frac > 0.5 in extreme "
              f"AR buckets) -- {len(df)} remain")
    assert 20 <= len(diagnostic) <= 40, \
        f"expected 20-40 diagnostic picks, got {len(diagnostic)} -- check {args.diagnostic_picks}"
    print(f"diagnostic_val: {len(diagnostic)} examples (excluded from val AND train)")

    remaining = df[~df["file_name"].isin(diagnostic)]
    val = stratified_val_split(remaining, args.val_fraction, args.seed)
    print(f"val: {len(val)} examples ({len(val) / len(df) * 100:.2f}% of total)")

    train = set(remaining["file_name"]) - val
    print(f"train: {len(train)} examples")

    assert not (diagnostic & val), "diagnostic/val overlap"
    assert not (diagnostic & train), "diagnostic/train overlap"
    assert not (val & train), "val/train overlap"

    write_manifest(os.path.join(args.out_root, "diagnostic_val.jsonl"), list(diagnostic), args.data_root)
    write_manifest(os.path.join(args.out_root, "val.jsonl"), list(val), args.data_root)
    write_manifest(os.path.join(args.out_root, "train.jsonl"), list(train), args.data_root)
    print(f"\nwrote manifests to {args.out_root}/{{diagnostic_val,val,train}}.jsonl")


if __name__ == "__main__":
    main()