import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import prepare_qualitative_appendix as appendix


RETAINED_CONTROLS = (
    "single_01_female_warrior",
    "single_02_wandering_knight",
    "single_03_floral_oracle",
    "single_04_stained_glass_saint",
    "single_07_crouched",
    "single_08_overhead_arms",
    "multi_01_gothic_duo",
    "multi_04_dense_group",
)
REMOVED_CONTROLS = {
    "single_05_editorial",
    "single_06_seated",
    "multi_02_mythic_duo",
    "multi_03_overlapping_pair",
}


class PrepareQualitativeAppendixTest(unittest.TestCase):
    def test_style_prompts_are_the_three_short_geometry_neutral_families(self):
        self.assertEqual(appendix.STYLE_PROMPTS, {
            "cinematic_fantasy_realism": (
                "Grounded cinematic fantasy character imagery, realistic materials, elegant costume design, "
                "naturalistic lighting, restrained atmosphere."
            ),
            "painterly_storybook_fantasy": (
                "Painterly fantasy illustration, warm palette, expressive brushwork, whimsical costume treatment, "
                "rich hand-painted texture."
            ),
            "classical_oil_painting": (
                "Classical oil painting, refined costume design, naturalistic skin tones, rich fabric textures, "
                "subtle chiaroscuro, restrained colors, finely rendered details."
            ),
        })
        self.assertNotIn("polished_modern_anime", appendix.STYLE_PROMPTS)
        self.assertNotIn("editorial_high_fashion_photography", appendix.STYLE_PROMPTS)
        for prompt in appendix.STYLE_PROMPTS.values():
            self.assertIsNone(appendix.POSE_WORDS.search(prompt), prompt)

    def test_controls_are_the_requested_eight_and_sources_exist(self):
        actual = tuple(row[0] for row in appendix.APPENDIX_CONTROLS)
        self.assertEqual(actual, RETAINED_CONTROLS)
        self.assertTrue(REMOVED_CONTROLS.isdisjoint(actual))
        self.assertEqual(len(appendix.APPENDIX_CONTROLS), 8)
        self.assertEqual(sum(row[2] == 1 for row in appendix.APPENDIX_CONTROLS), 6)
        self.assertEqual(sum(row[2] > 1 for row in appendix.APPENDIX_CONTROLS), 2)
        self.assertTrue(all(source.is_file() for _, _, _, source, _ in appendix.APPENDIX_CONTROLS))

    def test_actual_control_plan_has_24_unique_native_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            appendix_root = root / "appendix"
            with patch.object(appendix, "ROOT", root), \
                 patch.object(appendix, "APPENDIX", appendix_root):
                records = appendix.appendix_records()

        self.assertEqual(len(records), 24)
        self.assertEqual(tuple(dict.fromkeys(row["control_id"] for row in records)), RETAINED_CONTROLS)
        self.assertEqual({row["style_family"] for row in records}, set(appendix.STYLE_PROMPTS))
        self.assertEqual(len({row["output_path"] for row in records}), 24)
        self.assertTrue(all(row["width"] > 0 and row["height"] > 0
                            and row["width"] % 16 == 0 and row["height"] % 16 == 0
                            for row in records))

    def test_native_dimension_resolution_preserves_aspect_policy(self):
        cases = {
            (1001, 1001): (1024, 1024),
            (426, 640): (832, 1216),
            (612, 313): (1472, 704),
            (896, 1152): (896, 1152),
        }
        for source_size, expected in cases.items():
            with self.subTest(source_size=source_size):
                actual = appendix.resolve_native_appendix_dimensions(source_size)
                self.assertEqual(actual, expected)
                self.assertTrue(all(dimension > 0 and dimension % 16 == 0 for dimension in actual))

    def test_appendix_records_resolve_exactly_24_unique_rows_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_sizes = ((1001, 1001), (426, 640), (612, 313), (896, 1152))
            controls = []
            for index in range(8):
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
                self.assertTrue(all(Path(row["control_source_path"]).is_file() for row in records))

        self.assertEqual(len(records), 24)
        self.assertEqual({row["style_family"] for row in records}, set(appendix.STYLE_PROMPTS))
        self.assertEqual(len({row["output_path"] for row in records}), 24)
        self.assertTrue(all(row["width"] > 0 and row["height"] > 0
                            and row["width"] % 16 == 0 and row["height"] % 16 == 0
                            for row in records))
        self.assertTrue(all((row["width"], row["height"]) == appendix.resolve_native_appendix_dimensions(
                            (row["source_width"], row["source_height"])) for row in records))
        stale_row = next(row for row in records if row["id"] == "control_0__cinematic_fantasy_realism")
        self.assertEqual(stale_row["status"], "pending_cuda")
        self.assertIn("stale appendix output geometry", stale_row["failure"])

    def test_absent_outputs_are_pending_and_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls = []
            for index in range(8):
                source = root / f"source_{index}.png"
                Image.new("RGB", (896, 1152), "white").save(source)
                controls.append((f"control_{index}", "single", 1, source, "test"))
            appendix_root = root / "appendix"
            with patch.object(appendix, "ROOT", root), \
                 patch.object(appendix, "APPENDIX", appendix_root), \
                 patch.object(appendix, "APPENDIX_CONTROLS", tuple(controls)):
                records = appendix.appendix_records()

            self.assertEqual(sum(row["status"] == "pending_cuda" for row in records), 24)
            self.assertFalse((appendix_root / "generations").exists())
            self.assertEqual(len(list((appendix_root / "controls").glob("*.png"))), 8)

    def test_changed_prompt_sidecar_is_classified_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (896, 1152), "white").save(source)
            appendix_root = root / "appendix"
            output = appendix_root / "generations" / "control__cinematic_fantasy_realism.png"
            output.parent.mkdir(parents=True)
            Image.new("RGB", (896, 1152), "white").save(output)
            output.with_suffix(".json").write_text('{"prompt": "old prompt"}', encoding="utf-8")
            controls = tuple((f"control_{index}", "single", 1, source, "test") for index in range(8))
            controls = (("control", "single", 1, source, "test"), *controls[1:])
            with patch.object(appendix, "ROOT", root), \
                 patch.object(appendix, "APPENDIX", appendix_root), \
                 patch.object(appendix, "APPENDIX_CONTROLS", controls):
                records = appendix.appendix_records()

        row = next(row for row in records if row["id"] == "control__cinematic_fantasy_realism")
        self.assertEqual(row["status"], "pending_cuda")
        self.assertIn("stale appendix output provenance", row["failure"])
        self.assertEqual(row["source_evaluation_showcase_provenance"], "test")

    def test_appendix_sheet_has_control_plus_three_style_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            appendix_root = root / "appendix"
            control = appendix_root / "controls" / "control.png"
            control.parent.mkdir(parents=True)
            Image.new("RGB", (80, 160), "white").save(control)
            records = []
            for family in appendix.STYLE_PROMPTS:
                output = appendix_root / "generations" / f"control__{family}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (160, 80), "white").save(output)
                records.append({"control_id": "control", "style_family": family,
                                "copied_control_path": str(control.relative_to(root)),
                                "output_path": str(output.relative_to(root)), "status": "success"})
            captured = {}

            def fake_sheet(rows, destination, labels, title, thumb):
                captured.update(rows=rows, destination=destination, labels=labels, title=title, thumb=thumb)

            controls = (("control", "single", 1, control, "test"),)
            with patch.object(appendix, "ROOT", root), \
                 patch.object(appendix, "APPENDIX", appendix_root), \
                 patch.object(appendix, "APPENDIX_CONTROLS", controls), \
                 patch.object(appendix, "build_sheet", side_effect=fake_sheet):
                appendix.appendix_sheet(records)

        self.assertEqual(captured["labels"], ("control", *appendix.STYLE_PROMPTS))
        self.assertEqual(len(captured["rows"][0][1]), 4)
        self.assertEqual(captured["rows"][0][1][0].name, "control.png")


if __name__ == "__main__":
    unittest.main()
