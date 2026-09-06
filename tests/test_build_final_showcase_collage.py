"""Deterministic contract checks for the frozen public hero showcase."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_final_showcase_collage.py"
SPEC = importlib.util.spec_from_file_location("build_final_showcase_collage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalShowcaseContractTest(unittest.TestCase):
    def test_equal_weight_pair_geometry(self) -> None:
        pairs = [pair for row in MODULE.PAIR_ROWS for pair in row]
        self.assertEqual(
            [concept for concept, *_ in pairs],
            [
                "female_swordswoman_psychedelic",
                "fantasy_mage",
                "comic_fashion",
                "starry_night_painterly",
                "dark_fantasy_jester",
            ],
        )
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len({concept for concept, *_ in pairs}), 5)
        self.assertEqual(MODULE.TOP_SIDE_SIZE, (700, 1000))
        self.assertEqual(MODULE.BOTTOM_SIDE_SIZE, (1054, 800))
        for pair in pairs:
            condition_box, generation_box = MODULE.pair_boxes(pair)
            self.assertEqual(condition_box[2:], generation_box[2:])
            self.assertEqual(generation_box[0], condition_box[0] + condition_box[2] + MODULE.GUTTER)
            self.assertGreaterEqual(condition_box[2], 700)
            self.assertGreaterEqual(condition_box[3], 800)
        self.assertLess(
            (MODULE.BOTTOM_SIDE_SIZE[0] * MODULE.BOTTOM_SIDE_SIZE[1]) /
            (MODULE.TOP_SIDE_SIZE[0] * MODULE.TOP_SIDE_SIZE[1]),
            1.21,
        )
        self.assertGreater(MODULE.COLLAGE_SIZE[0], MODULE.COLLAGE_SIZE[1])

    def test_frozen_winner_selection_and_assets(self) -> None:
        contract = MODULE.build_contract()
        MODULE.verify_contract(contract)
        winners = {item["concept"]: item for item in contract["winners"]}
        self.assertEqual(
            {concept: item["selected_variant"] for concept, item in winners.items()},
            {
                "fantasy_mage": "hero_variant_b",
                "dark_fantasy_jester": "original_accepted",
                "comic_fashion": "hero_variant_b",
                "female_swordswoman_psychedelic": "original_accepted",
                "starry_night_painterly": "hero_variant_a",
            },
        )
        self.assertEqual(len(contract["metadata"]["alternates"]), 4)

    def test_prompting_guide_uses_frozen_winner_prompts(self) -> None:
        guide = (Path(__file__).resolve().parents[1] / "prompting.md").read_text(encoding="utf-8")
        for winner in MODULE.build_contract()["winners"]:
            self.assertIn(winner["prompt"], guide)


if __name__ == "__main__":
    unittest.main()
