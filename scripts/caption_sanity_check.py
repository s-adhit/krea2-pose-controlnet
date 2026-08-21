"""Caption/tokenizer sanity check. CPU-only, tokenizer-only (no model weights).

Mirrors TextConditioner's exact tokenization in trainer/train_control_lora.py:
PREFIX (34 tokens) + caption, truncated to max_length=512, + SUFFIX. Anything
whose caption alone would push past 512 tokens gets silently truncated by the
trainer -- this finds those cases before a training run does, plus basic data
hygiene (empty captions, exact duplicates, non-UTF8 issues).

Usage:
    python scripts/caption_sanity_check.py \
        --manifests data/manifests/train.jsonl,data/manifests/val.jsonl,data/manifests/diagnostic_val.jsonl \
        --max-length 512
"""

import argparse
import json
import os

import pandas as pd


PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def load_manifest_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests/train.jsonl,data/manifests/val.jsonl,data/manifests/diagnostic_val.jsonl")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out-dir", default="data/review/caption_check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for path in args.manifests.split(","):
        rows = load_manifest_rows(path.strip())
        split = os.path.splitext(os.path.basename(path.strip()))[0]
        for r in rows:
            r["_split"] = split
        all_rows.extend(rows)
        print(f"{split}: {len(rows)} rows")

    df = pd.DataFrame(all_rows)
    print(f"\n{len(df)} total rows across all manifests")

    # --- hygiene checks that don't need the tokenizer ---
    missing_text = df[df["text"].isna() | (df["text"].astype(str).str.strip() == "")]
    print(f"\nempty/missing captions: {len(missing_text)}")
    if len(missing_text):
        print(missing_text[["file_name", "_split"]].head(20).to_string(index=False))

    dupes = df[df.duplicated(subset=["text"], keep=False)].sort_values("text")
    print(f"\nexact-duplicate captions (same text, different image): {len(dupes)} rows "
          f"({dupes['text'].nunique()} distinct captions repeated)")
    if len(dupes):
        dupe_path = os.path.join(args.out_dir, "duplicate_captions.csv")
        dupes[["file_name", "_split", "text"]].to_csv(dupe_path, index=False)
        print(f"  wrote {dupe_path} -- worth a glance, could be legit (similar "
              f"scenes) or a captioning-pipeline stamp/template artifact")

    # --- tokenizer-based check, matching the real TextConditioner path ---
    print(f"\nloading tokenizer ({args.model_id})...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)

    prefix_len = len(tok(PREFIX)["input_ids"])
    print(f"measured PREFIX length: {prefix_len} tokens "
          f"(reference code assumes PREFIX_IDX=34 -- flag if these disagree)")

    texts = df["text"].fillna("").astype(str).tolist()
    full_texts = [PREFIX + t for t in texts]
    lengths = []
    batch = 64
    for i in range(0, len(full_texts), batch):
        enc = tok(full_texts[i:i + batch], truncation=False)
        lengths.extend(len(ids) - prefix_len for ids in enc["input_ids"])
    df["caption_token_len"] = lengths

    print(f"\ncaption token length (excluding PREFIX/SUFFIX):\n{df['caption_token_len'].describe()}")

    over = df[df["caption_token_len"] > args.max_length]
    print(f"\ncaptions exceeding max_length={args.max_length} "
          f"(would be SILENTLY TRUNCATED by the trainer): {len(over)}")
    if len(over):
        over_path = os.path.join(args.out_dir, "truncated_captions.csv")
        over.sort_values("caption_token_len", ascending=False)[
            ["file_name", "_split", "caption_token_len", "text"]
        ].to_csv(over_path, index=False)
        print(f"  wrote {over_path}")
        print(over.sort_values("caption_token_len", ascending=False)
              [["file_name", "_split", "caption_token_len"]].head(10).to_string(index=False))

    per_split = df.groupby("_split")["caption_token_len"].describe()
    print(f"\nper-split breakdown:\n{per_split}")

    df[["file_name", "_split", "caption_token_len"]].to_csv(
        os.path.join(args.out_dir, "all_caption_lengths.csv"), index=False
    )
    print(f"\nfull per-row token lengths written to "
          f"{args.out_dir}/all_caption_lengths.csv")


if __name__ == "__main__":
    main()