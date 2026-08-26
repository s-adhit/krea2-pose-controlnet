"""data.py — PoseBridge shard dataset with train/val split and
exclusion-list awareness (data/review/excluded_stems.csv)."""
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedLatentShardDataset(Dataset):
    """Read-only access to verified Gate-D ``split/split-*.pt`` archives.

    The compact index is derived from the archive headers/samples once and retains
    only shard number, in-shard offset, and bucket metadata.  Latents are loaded
    lazily from a one-shard cache; training never invokes VAE preprocessing.
    """
    def __init__(self, shard_root: str, split: str, *, text_conditioning_root: str | None = None) -> None:
        root = Path(shard_root)
        metadata_path = root / "shards.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or not metadata.get("complete"):
            raise ValueError(f"Latent root is not a verified complete shard set: {root}")
        self.root, self.split = root, split
        self.records: list[tuple[str, int, tuple[int, int], str]] = []
        for path in sorted((root / split).glob(f"{split}-*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format_version") != 1 or payload.get("split") != split:
                raise ValueError(f"Malformed {split} latent shard: {path}")
            for offset, sample in enumerate(payload.get("samples", ())):
                latent, control, text = sample.get("image_latent"), sample.get("control_latent"), sample.get("text")
                if not isinstance(latent, torch.Tensor) or not isinstance(control, torch.Tensor) or latent.shape != control.shape:
                    raise ValueError(f"Invalid paired latent at {path}:{offset}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing caption at {path}:{offset}")
                stem = sample.get("stem")
                if not isinstance(stem, str) or not stem:
                    raise ValueError(f"Missing stem at {path}:{offset}")
                self.records.append((str(path), offset, tuple(latent.shape[-2:]), stem))
        expected = metadata.get("expected_counts", {}).get(split)
        if expected is not None and len(self.records) != expected:
            raise ValueError(f"{split} shard count mismatch: got {len(self.records)}, expected {expected}")
        self._cached_path: str | None = None
        self._cached_samples: list[dict] | None = None
        self.text_conditioning = None
        if text_conditioning_root is not None:
            from pose_controlnet.text_conditioning import CachedTextConditioning
            self.text_conditioning = CachedTextConditioning(text_conditioning_root, split)
            stems = [record[3] for record in self.records]
            if set(stems) != set(self.text_conditioning.index) or len(stems) != len(self.text_conditioning.index):
                raise ValueError(f"Latent/text-conditioning stem identity mismatch for {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        path, offset, _, stem = self.records[index]
        if path != self._cached_path:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_path, self._cached_samples = path, payload["samples"]
        sample = self._cached_samples[offset]  # type: ignore[index]
        item = {"latent": sample["image_latent"], "control": sample["control_latent"], "prompt": sample["text"], "stem": stem}
        if self.text_conditioning is not None: item.update(self.text_conditioning.get(stem))
        return item


def load_prepared_sample(shard_root: str, split: str = "train",
                         shard_number: int = 0, sample_number: int = 0) -> dict:
    """Load and hard-check one sample from the persistent Gate-D shard format."""
    path = os.path.join(shard_root, split, f"{split}-{shard_number:05d}.pt")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"Could not load prepared latent shard {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Malformed prepared latent shard: {path}")
    if payload.get("split") != split:
        raise ValueError(f"Wrong split in prepared latent shard: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not 0 <= sample_number < len(samples):
        raise IndexError(f"Sample {sample_number} does not exist in {path}")
    sample = samples[sample_number]
    image = sample.get("image_latent")
    control = sample.get("control_latent")
    text = sample.get("text")
    if not isinstance(image, torch.Tensor) or not isinstance(control, torch.Tensor):
        raise ValueError(f"Missing latent tensors in {path} sample {sample_number}")
    if image.dtype != torch.float32 or control.dtype != torch.float32:
        raise ValueError(f"Latents are not float32 in {path} sample {sample_number}")
    if image.ndim != 3 or image.shape[0] != 16 or image.shape != control.shape:
        raise ValueError(f"Invalid paired latent shapes in {path} sample {sample_number}")
    if not torch.isfinite(image).all().item() or not torch.isfinite(control).all().item():
        raise ValueError(f"Non-finite latent in {path} sample {sample_number}")
    if control.abs().max().item() == 0.0:
        raise ValueError(f"Empty control latent in {path} sample {sample_number}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing caption in {path} sample {sample_number}")
    return {
        "latent": image,
        "control": control,
        "prompt": text,
        "stem": sample.get("stem"),
        "bucket": sample.get("bucket"),
        "shard_path": path,
        "sample_number": sample_number,
    }


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
    batch = {
        "latent": torch.stack([x["latent"] for x in items]),
        "control": torch.stack([x["control"] for x in items]),
        "prompts": [x["prompt"] for x in items],
    }
    if "context" in items[0]:
        if any("context" not in item or "mask" not in item for item in items): raise ValueError("Mixed cached/non-cached text batch")
        contexts, masks = [x["context"] for x in items], [x["mask"] for x in items]
        length = max(context.shape[0] for context in contexts)
        batch["context"] = torch.stack([torch.nn.functional.pad(context, (0, 0, 0, 0, 0, length - context.shape[0])) for context in contexts])
        batch["text_mask"] = torch.stack([torch.nn.functional.pad(mask, (0, length - mask.shape[0])) for mask in masks])
    return batch
