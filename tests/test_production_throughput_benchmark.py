import unittest

from pose_controlnet.throughput_benchmark import (
    LOCKED_EFFECTIVE_BATCH,
    ThroughputBenchmarkRecipe,
    projected_runtime,
    validate_benchmark_result,
)


class ProductionThroughputBenchmarkTests(unittest.TestCase):
    def test_locked_recipe_preserves_effective_batch(self):
        recipe = ThroughputBenchmarkRecipe(microbatch_size=4, gradient_accumulation_steps=8)
        recipe.validate()
        self.assertEqual(recipe.effective_batch_size, LOCKED_EFFECTIVE_BATCH)

    def test_recipe_rejects_semantic_batch_or_worker_changes(self):
        with self.assertRaisesRegex(ValueError, "effective batch"):
            ThroughputBenchmarkRecipe(microbatch_size=2, gradient_accumulation_steps=8).validate()
        with self.assertRaisesRegex(ValueError, "persistent_workers"):
            ThroughputBenchmarkRecipe(persistent_workers=True).validate()
        with self.assertRaisesRegex(ValueError, "resolution policy"):
            ThroughputBenchmarkRecipe(resolution_policy="native").validate()

    def test_runtime_projection_uses_actual_batch_and_dataset(self):
        rows = projected_runtime(seconds_per_optimizer_step=2.0, effective_batch_size=32, training_samples=160)
        self.assertEqual(rows[0]["optimizer_steps"], 1000)
        self.assertEqual(rows[0]["sample_presentations"], 32000)
        self.assertEqual(rows[0]["dataset_equivalent_passes"], 200.0)
        self.assertEqual(rows[0]["wall_hours"], 2000 / 3600)

    def test_result_schema_is_fail_closed(self):
        recipe = ThroughputBenchmarkRecipe().asdict()
        result = {
            "recipe": recipe, "trainable_parameter_names": ["first.weight"], "trainable_parameter_count": 1,
            "forward_seconds_mean": 1.0, "backward_seconds_mean": 1.0, "optimizer_seconds_mean": 1.0,
            "optimizer_step_seconds_mean": 3.0, "samples_per_second": 10.0,
            "effective_samples_per_second": 10.0, "data_wait_seconds_mean": 0.1,
            "cuda_allocated_bytes": 1, "cuda_peak_allocated_bytes": 2, "pose_active_fraction": 0.1,
            "pose_active_microbatch_fraction": 0.1, "runtime_projection": [],
        }
        validate_benchmark_result(result)
        result.pop("optimizer_seconds_mean")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_benchmark_result(result)


if __name__ == "__main__":
    unittest.main()
