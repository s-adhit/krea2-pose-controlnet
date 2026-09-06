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
