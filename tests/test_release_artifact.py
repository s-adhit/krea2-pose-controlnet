import tempfile
import unittest
from pathlib import Path

import torch

from pose_controlnet.release_artifact import (
    ReleaseArtifactError,
    interpolate_model_tensors_fp32,
    load_release_artifact,
    save_release_artifact,
)


class ReleaseArtifactTest(unittest.TestCase):
    def _state(self, tensor: torch.Tensor, **extra):
        return {"model": {"first.weight": tensor}, **extra}

    def test_key_mismatch_is_rejected(self):
        parent = self._state(torch.tensor([1.0]))
        finish = {"model": {"first.bias": torch.tensor([2.0])}}
        with self.assertRaisesRegex(ReleaseArtifactError, "exact matching trainable keys"):
            interpolate_model_tensors_fp32(parent, finish, 0.25)

    def test_shape_mismatch_is_rejected(self):
        parent = self._state(torch.tensor([1.0]))
        finish = self._state(torch.tensor([[2.0]]))
        with self.assertRaisesRegex(ReleaseArtifactError, "shape mismatch"):
            interpolate_model_tensors_fp32(parent, finish, 0.25)

    def test_exact_alpha_interpolation_is_saved_as_float32(self):
        parent = self._state(torch.tensor([1.0, -3.0], dtype=torch.float32))
        finish = self._state(torch.tensor([5.0, 9.0], dtype=torch.float32))
        mixed = interpolate_model_tensors_fp32(parent, finish, 0.25)
        self.assertEqual(mixed["first.weight"].dtype, torch.float32)
        self.assertTrue(torch.equal(mixed["first.weight"], torch.tensor([2.0, 0.0], dtype=torch.float32)))

    def test_non_model_training_state_is_excluded_from_release_artifact(self):
        parent = self._state(torch.tensor([1.0]), optimizer={"momentum": torch.tensor([99.0])},
                             scheduler={"step_count": 4000}, rng={"torch": "parent"}, global_step=4000)
        finish = self._state(torch.tensor([5.0]), optimizer={"momentum": torch.tensor([-99.0])},
                             scheduler={"step_count": 4300}, rng={"torch": "finish"}, global_step=4300)
        mixed = interpolate_model_tensors_fp32(parent, finish, 0.25)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "mix.safetensors"
            save_release_artifact(
                artifact, mixed, release_id="release", candidate="mix-025", alpha=0.25,
                raw_checkpoint="raw.safetensors", release_contract_sha256="a" * 64,
            )
            reloaded = load_release_artifact(artifact)
        self.assertEqual(set(reloaded), {"format_version", "release", "config", "model"})
        self.assertEqual(set(reloaded["model"]), {"first.weight"})
        self.assertTrue(torch.equal(reloaded["model"]["first.weight"], torch.tensor([2.0])))

    def test_serialization_is_byte_deterministic(self):
        model = {"first.weight": torch.tensor([2.0], dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.safetensors", root / "second.safetensors"
            kwargs = {"release_id": "release", "candidate": "mix-025", "alpha": 0.25,
                      "raw_checkpoint": "raw.safetensors", "release_contract_sha256": "a" * 64}
            save_release_artifact(first, model, **kwargs)
            save_release_artifact(second, model, **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
