"""data.py — PoseBridge shard dataset with train/val split and
exclusion-list awareness (data/review/excluded_stems.csv)."""
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseShardDataset(Dataset):
    def __init__(self, shard_dir: str, split: str = "train",
                 exclude_stems_path: str | None = None):
        excluded = set()
        if exclude_stems_path and os.path.exists(exclude_stems_path):
            with open(exclude_stems_path) as f:
                excluded = {line.strip() for line in f if line.strip()}
            print(f"[data] loaded {len(excluded)} excluded stems from {exclude_stems_path}")

        self.records = []
        for idx_file in sorted(glob.glob(os.path.join(shard_dir, "shard*", "index.jsonl"))):
            with open(idx_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("split", "train") != split:
                        continue
                    if rec.get("stem") in excluded:
                        continue
                    self.records.append(rec)
        self.shard_dir = shard_dir
        print(f"[data] split={split}: {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        d = np.load(os.path.join(self.shard_dir, rec["file"]))
        return {
            "latent": torch.from_numpy(d["latent"].astype(np.float32)),
            "control": torch.from_numpy(d["control"].astype(np.float32)),
            "prompt": str(d["prompt"]),
        }


class BucketBatchSampler:
    def __init__(self, records, batch_size: int, seed: int):
        self.by_bucket: dict[tuple, list[int]] = {}
        for i, rec in enumerate(records):
            self.by_bucket.setdefault(tuple(rec["bucket"]), []).append(i)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for idxs in self.by_bucket.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            for i in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                batches.append(idxs[i: i + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(len(v) // self.batch_size for v in self.by_bucket.values())


def collate(items):
    return {
        "latent": torch.stack([x["latent"] for x in items]),
        "control": torch.stack([x["control"] for x in items]),
        "prompts": [x["prompt"] for x in items],
    }