import unittest

from scripts import final_showcase_batch1 as showcase


class FinalShowcaseBatch1Test(unittest.TestCase):
    def test_frozen_manifest_count_order_native_and_release_settings(self):
        manifest = showcase.load_manifest()
        rows = showcase.validate_manifest(manifest, check_files=False)
        self.assertEqual(len(rows), 12)
        self.assertEqual(tuple(row["id"] for row in rows), showcase.EXPECTED_ORDER)
        self.assertTrue(all(row["geometry"] == showcase.NATIVE_GEOMETRY for row in rows))
        self.assertTrue(all(row["style_lora"] is None for row in rows))
        self.assertTrue(all(row["candidate"] == "mix-025" and row["control_scale"] == 1.0 for row in rows))
        self.assertTrue(all(row["turbo"] == showcase.EXPECTED_TURBO for row in rows))
        prompts = {}
        for row in rows:
            prompts.setdefault(row["concept"], row["prompt"])
            self.assertEqual(prompts[row["concept"]], row["prompt"])

