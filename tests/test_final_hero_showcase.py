import copy
import unittest

from scripts import final_hero_showcase as hero


class FinalHeroShowcaseTest(unittest.TestCase):
    def test_frozen_hero_manifest_resolves_the_five_accepted_winners(self):
        manifest = hero.load_manifest()
        rows = hero.validate_manifest(manifest)
        winners = hero.resolve_winners(manifest, check_files=False)

        self.assertEqual(tuple(row["id"] for row in rows), tuple(item[0] for item in hero.EXPECTED_WINNERS))
        self.assertEqual(tuple(winner["id"] for winner in winners), tuple(item[0] for item in hero.EXPECTED_WINNERS))
        self.assertEqual([variant["seed"] for winner in winners for variant in winner["hero_variants"]], list(hero.EXPECTED_HERO_SEEDS))
        self.assertTrue(all(variant["seed"] not in {winner["original_seed"] for winner in winners}
                            for winner in winners for variant in winner["hero_variants"]))

    def test_resolved_winners_are_locked_to_the_native_no_style_release_contract(self):
        manifest = hero.load_manifest()
        winners = hero.resolve_winners(manifest, check_files=False)

        self.assertTrue(all(winner["native_bucket"][0] % 16 == 0 and winner["native_bucket"][1] % 16 == 0
                            for winner in winners))
        self.assertEqual(hero.EXPECTED_RELEASE["sha256"], hero.sha256(hero.ROOT / hero.EXPECTED_RELEASE["path"]))
        self.assertTrue(all(variant["output_path"].startswith(str(hero.EXPECTED_OUTPUT_ROOT))
                            for winner in winners for variant in winner["hero_variants"]))

    def test_manifest_seed_drift_fails_closed(self):
        manifest = copy.deepcopy(hero.load_manifest())
        manifest["rows"][0]["hero_variants"][0]["seed"] = 7194308251

        with self.assertRaises(hero.HeroShowcaseError):
            hero.validate_manifest(manifest)
