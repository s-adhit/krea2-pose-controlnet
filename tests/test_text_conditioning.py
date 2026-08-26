import json
import tempfile
import unittest
from pathlib import Path

import torch

import train
from pose_controlnet.data import collate
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.text_conditioning import (
    CachedTextConditioning,
    FORMAT_VERSION,
    METADATA_NAME,
    TextConditioningError,
    _validate_entry,
    compact_valid_conditioning,
)


def _entry(stem: str, length: int = 2) -> dict:
    return {"stem": stem, "context": torch.ones(length, 12, 4, dtype=torch.bfloat16), "mask": torch.ones(length, dtype=torch.bool)}


class _BatchEncoding(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    def __init__(self, conditioner): self.conditioner = conditioner

    def __call__(self, texts, **_kwargs):
        text = texts[0]
        if text == self.conditioner.SUFFIX:
            ids = [100, 101]
        else:
            caption = text[len(self.conditioner.PREFIX):]
            ids = [1] + [2] * (1 if caption == "short" else 4)
        values = torch.tensor([ids], dtype=torch.long)
        return _BatchEncoding(input_ids=values, attention_mask=torch.ones_like(values))


class _Qwen(torch.nn.Module):
    def forward(self, *, input_ids, attention_mask, output_hidden_states):
        del attention_mask, output_hidden_states
        hidden = input_ids.unsqueeze(-1).float()
        return type("Output", (), {"hidden_states": tuple(hidden for _ in range(36))})()


def _fake_conditioner() -> PoseTextConditioner:
    conditioner = PoseTextConditioner.__new__(PoseTextConditioner)
    torch.nn.Module.__init__(conditioner)
    conditioner.PREFIX_IDX = 1
    conditioner.max_length = 32
    conditioner.device = "cpu"
    conditioner.qwen = _Qwen()
    conditioner.tokenizer = _Tokenizer(conditioner)
    return conditioner


class CachedTextConditioningTest(unittest.TestCase):
    def test_pose_text_conditioner_independent_batching_moves_suffix_next_to_prompt(self):
        conditioner = _fake_conditioner()
        batched_context, batched_mask = conditioner(["short", "long"])
        short_context, short_mask = conditioner(["short"])
        long_context, long_mask = conditioner(["long"])
        short_cached = compact_valid_conditioning(batched_context, batched_mask, 0)
        long_cached = compact_valid_conditioning(batched_context, batched_mask, 1)

        self.assertTrue(torch.equal(short_cached["context"], short_context[0][short_mask[0]]))
        self.assertTrue(torch.equal(long_cached["context"], long_context[0][long_mask[0]]))
        self.assertTrue(short_cached["mask"].all())
        self.assertTrue(torch.equal(short_cached["context"][-2:, :, 0], torch.tensor([[100.], [101.]]).expand(2, 12)))
        self.assertFalse(batched_mask[0, short_cached["mask"].shape[0]:].any())

    def test_boolean_compaction_preserves_suffix_and_matches_independent_online(self):
        # This is the real regression layout from the old mixed-length encoder:
        # short prompt tokens, internal padding, then valid suffix tokens.
        contexts = torch.arange(2 * 8 * 12 * 4, dtype=torch.float32).reshape(2, 8, 12, 4).to(torch.bfloat16)
        masks = torch.tensor([
            [True, True, False, False, False, True, True, True],
            [True, True, True, True, True, True, True, True],
        ])
        short_cached = compact_valid_conditioning(contexts, masks, 0)
        long_cached = compact_valid_conditioning(contexts, masks, 1)
        short_online = contexts[0][masks[0]]
        long_online = contexts[1][masks[1]]

        self.assertTrue(short_cached["mask"].all())
        self.assertEqual(short_cached["context"].shape[0], 5)
        self.assertTrue(torch.equal(short_cached["context"], short_online))
        self.assertTrue(torch.equal(long_cached["context"], long_online))
        # The final three positions are suffix tokens, not truncated padding.
        self.assertTrue(torch.equal(short_cached["context"][-3:], contexts[0, 5:8]))
        self.assertEqual((short_cached["context"].float() - short_online.float()).abs().max().item(), 0.0)
        self.assertEqual((long_cached["context"].float() - long_online.float()).abs().max().item(), 0.0)

    def test_unconditional_uses_the_same_boolean_compaction(self):
        contexts = torch.arange(6 * 12 * 4, dtype=torch.float32).reshape(1, 6, 12, 4).to(torch.bfloat16)
        masks = torch.tensor([[True, False, False, True, True, True]])
        unconditional = compact_valid_conditioning(contexts, masks, 0)
        self.assertTrue(unconditional["mask"].all())
        self.assertTrue(torch.equal(unconditional["context"], contexts[0][masks[0]]))

    def test_generated_normal_entry_retains_stem_and_validates(self):
        contexts = torch.ones(1, 3, 12, 4, dtype=torch.bfloat16)
        masks = torch.tensor([[True, False, True]])
        stem = "coco_100098_193288"
        entry = {
            "stem": stem,
            **{
                key: value.detach().cpu().to(torch.bfloat16 if key == "context" else torch.bool).contiguous()
                for key, value in compact_valid_conditioning(contexts, masks, 0).items()
            },
        }

        observed_stem, dimensions = _validate_entry(entry, path="train-00000.pt", expected_stem=stem)

        self.assertEqual(observed_stem, stem)
        self.assertEqual(dimensions, (12, 4))
        self.assertEqual(entry["context"].dtype, torch.bfloat16)
        self.assertEqual(entry["mask"].dtype, torch.bool)

    def test_dynamic_right_padding_preserves_compacted_valid_content(self):
        contexts = torch.arange(2 * 8 * 12 * 4, dtype=torch.float32).reshape(2, 8, 12, 4).to(torch.bfloat16)
        masks = torch.tensor([
            [True, True, False, False, False, True, True, True],
            [True, True, True, True, True, True, True, True],
        ])
        short = compact_valid_conditioning(contexts, masks, 0)
        long = compact_valid_conditioning(contexts, masks, 1)
        batch = collate([
            {"latent": torch.ones(1, 2, 2), "control": torch.ones(1, 2, 2), "prompt": "short", **short},
            {"latent": torch.ones(1, 2, 2), "control": torch.ones(1, 2, 2), "prompt": "long", **long},
        ])
        self.assertTrue(torch.equal(batch["context"][0, :short["context"].shape[0]], short["context"]))
        self.assertTrue(torch.equal(batch["text_mask"][0, :short["mask"].shape[0]], short["mask"]))
        self.assertFalse(batch["text_mask"][0, short["mask"].shape[0]:].any())
        self.assertTrue(torch.equal(batch["context"][1], long["context"]))

    def test_cached_loader_collate_and_seeded_unconditional_dropout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / METADATA_NAME).write_text(json.dumps({"format_version": FORMAT_VERSION, "complete": True,
                "context_dimensions": [12, 4], "expected_counts": {"train": 1, "val": 1, "diagnostic_val": 1},
                "total_samples": 3, "shard_samples": 64, "context_dtype": "bfloat16", "mask_dtype": "bool"}))
            (root / "train").mkdir()
            torch.save({"format_version": FORMAT_VERSION, "split": "train", "samples": [_entry("stem")]}, root / "train/train-00000.pt")
            torch.save({"format_version": FORMAT_VERSION, "context": torch.full((1, 12, 4), 3.0, dtype=torch.bfloat16), "mask": torch.ones(1, dtype=torch.bool)}, root / "unconditional.pt")
            cache = CachedTextConditioning(root, "train")
            item = cache.get("stem")
            batch = collate([{"latent": torch.ones(1, 2, 2), "control": torch.ones(1, 2, 2), "prompt": "caption", **item}])
            train.apply_cached_caption_dropout(batch, cache.unconditional, 1.0, 42, 0)
            self.assertEqual(batch["context"].dtype, torch.bfloat16)
            self.assertEqual(batch["text_mask"].dtype, torch.bool)
            self.assertEqual(batch["context"].shape, (1, 1, 12, 4))
            self.assertEqual(batch["context"][0, 0, 0, 0].item(), 3.0)

    def test_cached_loader_rejects_nonfinite_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / METADATA_NAME).write_text(json.dumps({"format_version": FORMAT_VERSION, "complete": True,
                "context_dimensions": [12, 4], "expected_counts": {}, "total_samples": 0, "shard_samples": 64,
                "context_dtype": "bfloat16", "mask_dtype": "bool"}))
            (root / "train").mkdir()
            bad = _entry("stem"); bad["context"][0, 0, 0] = float("nan")
            torch.save({"format_version": FORMAT_VERSION, "split": "train", "samples": [bad]}, root / "train/train-00000.pt")
            torch.save({"format_version": FORMAT_VERSION, "context": torch.ones(1, 12, 4, dtype=torch.bfloat16), "mask": torch.ones(1, dtype=torch.bool)}, root / "unconditional.pt")
            with self.assertRaises(TextConditioningError): CachedTextConditioning(root, "train")

    def test_cached_loader_rejects_old_format_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / METADATA_NAME).write_text(json.dumps({"format_version": FORMAT_VERSION - 1, "complete": True,
                "context_dimensions": [12, 4], "expected_counts": {}, "total_samples": 0, "shard_samples": 64,
                "context_dtype": "bfloat16", "mask_dtype": "bool"}))
            with self.assertRaises(TextConditioningError): CachedTextConditioning(root, "train")


if __name__ == "__main__": unittest.main()
