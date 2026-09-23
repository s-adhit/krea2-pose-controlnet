import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import prepare_qualitative_appendix as appendix


class PrepareQualitativeAppendixTest(unittest.TestCase):
    def test_native_dimension_resolution_preserves_aspect_policy(self):
        cases = {
            # Square and the two orientations deliberately start off the
            # VAE/model 16-pixel grid.  The result is the shared native bucket,
            # not a forced square or the alternate dynamic-768 bucket.
            (1001, 1001): (1024, 1024),
            (426, 640): (832, 1216),
            (612, 313): (1472, 704),
            # A control already at a frozen native bucket stays unchanged.
            (896, 1152): (896, 1152),
        }
        for source_size, expected in cases.items():
            with self.subTest(source_size=source_size):
                actual = appendix.resolve_native_appendix_dimensions(source_size)
                self.assertEqual(actual, expected)
                self.assertTrue(all(dimension > 0 and dimension % 16 == 0 for dimension in actual))

    def test_appendix_records_resolve_all_rows_without_loading_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_sizes = ((1001, 1001), (426, 640), (612, 313), (896, 1152))
            controls = []
            for index in range(12):
                source = root / f"source_{index}.png"
                Image.new("RGB", source_sizes[index % len(source_sizes)], "white").save(source)
                controls.append((f"control_{index}", "single", 1, source, "test"))
            appendix_root = root / "appendix"
            stale = appendix_root / "generations" / "control_0__cinematic_fantasy_realism.png"
            stale.parent.mkdir(parents=True)
            Image.new("RGB", (640, 640), "white").save(stale)
            with patch.object(appendix, "ROOT", root), \
                 patch.object(appendix, "APPENDIX", appendix_root), \
                 patch.object(appendix, "APPENDIX_CONTROLS", tuple(controls)):
                records = appendix.appendix_records()

        self.assertEqual(len(records), 48)
        self.assertTrue(all(row["width"] > 0 and row["height"] > 0
                            and row["width"] % 16 == 0 and row["height"] % 16 == 0
                            for row in records))
        self.assertTrue(all((row["width"], row["height"]) == appendix.resolve_native_appendix_dimensions(
                            (row["source_width"], row["source_height"])) for row in records))
        stale_row = next(row for row in records if row["id"] == "control_0__cinematic_fantasy_realism")
        self.assertEqual(stale_row["status"], "pending_cuda")
        self.assertIn("stale appendix output geometry", stale_row["failure"])


if __name__ == "__main__":
    unittest.main()
