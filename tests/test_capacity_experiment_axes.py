import unittest

from pose_controlnet.capacity_resolution import _validate_geometry
from pose_controlnet.overfit_capacity import (
    CapacityScientificConfig, RESOLUTION_768_BUCKETS, capacity_experiment_name,
    canonical_resolution_policy, validate_capacity_scientific_config,
)


class CapacityExperimentAxesTests(unittest.TestCase):
    def test_768_policy_is_fixed_aspect_preserving_and_named(self):
        self.assertEqual(canonical_resolution_policy("current"), "native")
        self.assertEqual(RESOLUTION_768_BUCKETS, ((768, 768), (704, 896), (896, 704), (640, 960), (960, 640), (576, 1024), (1024, 576), (512, 1152), (1152, 512)))
        self.assertEqual(capacity_experiment_name("mixed32", "768", "none", 0), "overfit32-mixed-r64-mse-res768")
        self.assertEqual(capacity_experiment_name("mixed32", "768", "normalized_coordinate_huber", 1e-5), "overfit32-mixed-r64-coord-l1e-5-res768")

    def test_geometry_rejects_stale_native_or_unpaired_shapes(self):
        geometry = {"source_size": [1000, 700], "resized_size": [1280, 896], "crop_box": [0, 0, 768, 768], "bucket": [768, 768], "latent_size": [96, 96]}
        _validate_geometry(geometry, stem="x")
        geometry["latent_size"] = [128, 128]
        with self.assertRaisesRegex(ValueError, "latent geometry"):
            _validate_geometry(geometry, stem="x")

    def test_pose_axes_fail_closed_and_preserve_baseline(self):
        baseline = validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32"))
        self.assertEqual((baseline.pose_loss, baseline.lambda_pose, baseline.forced_pose_exposure_probability), ("none", 0.0, 0.0))
        selected = validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32", resolution="768", pose_loss="normalized_coordinate_huber", lambda_pose=1e-5, pose_timestep_min=.1, pose_timestep_max=.2))
        self.assertEqual(selected.pose_loss, "normalized_coordinate_huber")
        with self.assertRaises(ValueError): validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32", lambda_pose=1e-5))
        with self.assertRaises(ValueError): validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32", pose_loss="normalized_coordinate_huber", lambda_pose=0))
        with self.assertRaises(ValueError): canonical_resolution_policy("512")


if __name__ == "__main__": unittest.main()
