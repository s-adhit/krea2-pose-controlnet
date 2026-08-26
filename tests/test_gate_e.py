import tempfile
import types
import unittest
from pathlib import Path

import torch

from pose_controlnet.data import load_prepared_sample
from pose_controlnet.diffusion import make_flow_pair, patchify_and_position, sample_flow_timestep
from pose_controlnet.model import LORA_TARGETS, POSE_CONFIG

from base_model.k2_lora import ControlInputLayer, LoRALinear


class GateEUnitTests(unittest.TestCase):
    def test_official_architecture_and_lora_contract(self) -> None:
        self.assertEqual(POSE_CONFIG.features, 6144)
        self.assertEqual(POSE_CONFIG.layers, 28)
        self.assertEqual(POSE_CONFIG.heads, 48)
        self.assertEqual(POSE_CONFIG.kvheads, 12)
        self.assertEqual(POSE_CONFIG.features // POSE_CONFIG.heads, 128)
        self.assertEqual(POSE_CONFIG.channels * POSE_CONFIG.patch**2, 64)
        self.assertEqual(LORA_TARGETS, (
            "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
            "mlp.gate", "mlp.up", "mlp.down",
        ))
        self.assertEqual(POSE_CONFIG.layers * len(LORA_TARGETS), 224)
        mlp_features = 16384
        per_block = 64 * (
            (6144 + 6144) * 3 + (6144 + 1536) * 2
            + (6144 + mlp_features) * 2 + (mlp_features + 6144)
        )
        control_input = 6144 * 128 + 6144
        self.assertEqual(per_block * 28 + control_input, 215_488_512)

    def test_control_input_copies_image_half_and_preserves_tokens(self) -> None:
        torch.manual_seed(42)
        pretrained = torch.nn.Linear(64, 32)
        original_weight = pretrained.weight.detach().float().clone()
        original_bias = pretrained.bias.detach().float().clone()
        layer = ControlInputLayer(pretrained)

        self.assertEqual(tuple(layer.weight.shape), (32, 128))
        self.assertTrue(torch.equal(layer.weight[:, :64], original_weight))
        self.assertEqual(torch.count_nonzero(layer.weight[:, 64:]).item(), 0)
        self.assertTrue(torch.equal(layer.bias, original_bias))

        image = torch.randn(2, 7, 64)
        control = torch.randn_like(image)
        real_output = layer(torch.cat((image, control), dim=-1))
        zero_output = layer(torch.cat((image, torch.zeros_like(control)), dim=-1))
        self.assertEqual(tuple(real_output.shape), (2, 7, 32))
        self.assertTrue(torch.equal(real_output, pretrained(image)))
        self.assertTrue(torch.equal(real_output, zero_output))

    def test_lora_rank_and_zero_impact_initialization(self) -> None:
        torch.manual_seed(42)
        base = torch.nn.Linear(16, 12, bias=False).requires_grad_(False)
        layer = LoRALinear(base, rank=64, alpha=64)
        inputs = torch.randn(3, 16)
        self.assertEqual(tuple(layer.A.shape), (64, 16))
        self.assertEqual(tuple(layer.B.shape), (12, 64))
        self.assertTrue(torch.equal(layer(inputs), base(inputs)))
        layer(inputs).square().mean().backward()
        self.assertEqual(torch.count_nonzero(layer.A.grad).item(), 0)
        self.assertGreater(torch.linalg.vector_norm(layer.B.grad).item(), 0)
        self.assertIsNone(base.weight.grad)

    def test_shifted_logistic_normal_flow_pair(self) -> None:
        cfg = types.SimpleNamespace(mu_x1=256.0, mu_y1=0.5, mu_x2=6400.0, mu_y2=1.15)
        generator = torch.Generator().manual_seed(42)
        timestep = sample_flow_timestep(2, 3952, cfg, "cpu", generator)
        self.assertTrue(torch.all((timestep > 0) & (timestep < 1)))
        clean = torch.randn(2, 16, 8, 8)
        noise = torch.randn_like(clean)
        noisy, target = make_flow_pair(clean, noise, timestep)
        expected = timestep[:, None, None, None] * noise + (1 - timestep[:, None, None, None]) * clean
        self.assertTrue(torch.equal(noisy, expected))
        self.assertTrue(torch.equal(target, noise - clean))

        text_mask = torch.ones(2, 3, dtype=torch.bool)
        tokens, pos, mask = patchify_and_position(noisy, 3, patch=2, txt_mask=text_mask)
        self.assertEqual(tuple(tokens.shape), (2, 16, 64))
        self.assertEqual(tuple(pos.shape), (2, 19, 3))
        self.assertEqual(tuple(mask.shape), (2, 19))

    def test_one_sample_loader_reads_gate_d_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "train"
            split_dir.mkdir()
            image = torch.randn(16, 8, 12, dtype=torch.float32)
            control = torch.randn_like(image)
            torch.save({
                "format_version": 1,
                "split": "train",
                "samples": [{
                    "stem": "sample",
                    "bucket": [96, 64],
                    "text": "a caption",
                    "image_latent": image,
                    "control_latent": control,
                }],
            }, split_dir / "train-00000.pt")
            sample = load_prepared_sample(str(root))
            self.assertTrue(torch.equal(sample["latent"], image))
            self.assertTrue(torch.equal(sample["control"], control))
            self.assertEqual(sample["prompt"], "a caption")


if __name__ == "__main__":
    unittest.main()
