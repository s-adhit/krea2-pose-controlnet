import types
import unittest

import torch
from PIL import Image

from pose_controlnet.vae_preprocessing import (
    VAEPreprocessingError,
    encode_preprocessed_pair,
    normalize_qwen_latents,
    pil_to_qwen_vae_tensor,
    tensor_report,
)


class _FakeVAE:
    config = types.SimpleNamespace(z_dim=2, latents_mean=[1.0, -2.0], latents_std=[2.0, 4.0])


class _FakeDistribution:
    def __init__(self, latent: torch.Tensor) -> None:
        self.latent = latent

    def sample(self, generator=None) -> torch.Tensor:
        return self.latent


class _EncodingFakeVAE:
    config = types.SimpleNamespace(z_dim=2, latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor):
        self.inputs.append(pixels)
        value = pixels.mean()
        latent = torch.full((1, 2, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8), value)
        return types.SimpleNamespace(latent_dist=_FakeDistribution(latent))


class VAEPreprocessingTest(unittest.TestCase):
    def test_rgb_pil_conversion_preserves_qwen_image_layout_and_range(self) -> None:
        image = Image.new("RGB", (2, 1))
        image.putdata([(0, 127, 255), (255, 0, 127)])
        tensor = pil_to_qwen_vae_tensor(image)
        self.assertEqual(tensor.shape, (1, 3, 1, 1, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(float(tensor.min()), -1.0)
        self.assertEqual(float(tensor.max()), 1.0)
        self.assertAlmostEqual(float(tensor[0, 1, 0, 0, 0]), 127 / 127.5 - 1.0)

    def test_per_channel_qwen_normalization(self) -> None:
        raw = torch.tensor([[[[[3.0]]], [[[6.0]]]]])
        normalized = normalize_qwen_latents(raw, _FakeVAE())
        self.assertTrue(torch.equal(normalized, torch.tensor([[[[[1.0]]], [[[2.0]]]]])))

    def test_invalid_latent_statistics_and_nonfinite_latents_fail_loudly(self) -> None:
        invalid = _FakeVAE()
        invalid.config = types.SimpleNamespace(z_dim=2, latents_mean=[0.0], latents_std=[1.0])
        raw = torch.zeros(1, 2, 1, 1, 1)
        with self.assertRaisesRegex(VAEPreprocessingError, "statistics"):
            normalize_qwen_latents(raw, invalid)
        raw[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(VAEPreprocessingError, "NaN or Inf"):
            normalize_qwen_latents(raw, _FakeVAE())

    def test_tensor_report_is_compact_and_finite(self) -> None:
        report = tensor_report(torch.tensor([[-3.0, 4.0]]))
        self.assertEqual(report["shape"], [1, 2])
        self.assertEqual(report["dtype"], "torch.float32")
        self.assertTrue(report["finite"])
        self.assertEqual(report["min"], -3.0)
        self.assertEqual(report["max"], 4.0)

    def test_paired_encode_uses_one_frame_layout_and_returns_matched_clean_latents(self) -> None:
        pair = types.SimpleNamespace(
            rgb=Image.new("RGB", (16, 24), (255, 0, 0)),
            control=Image.new("RGB", (16, 24), (0, 255, 0)),
        )
        vae = _EncodingFakeVAE()
        encoded = encode_preprocessed_pair(vae, pair, device="cpu", dtype=torch.float32)
        self.assertEqual([tensor.shape for tensor in vae.inputs], [(1, 3, 1, 24, 16)] * 2)
        self.assertEqual(encoded.latent.shape, (2, 3, 2))
        self.assertEqual(encoded.control.shape, encoded.latent.shape)
        self.assertGreater(float(encoded.control.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
