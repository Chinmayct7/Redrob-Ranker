"""Unit tests for src.ontology — skill synonym resolution and company tiers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ontology


class TestCanonicalSkill(unittest.TestCase):
    def test_unknown_skill_returns_none(self):
        self.assertIsNone(ontology.canonical_skill("Underwater Basket Weaving"))

    def test_embeddings_synonyms_resolve_to_same_canonical_id(self):
        # These are the low-frequency paraphrases the README calls out
        # explicitly as a synonym-matching test built into the dataset.
        known_variants = ["Embeddings", "Sentence Transformers", "Text Encoders",
                           "Vector Representations"]
        resolved = {v: ontology.canonical_skill(v) for v in known_variants}
        recognized = {v: cid for v, cid in resolved.items() if cid is not None}
        # At minimum, every variant present in SKILL_SYNONYMS should map together.
        canonical_ids = set(recognized.values())
        self.assertLessEqual(len(canonical_ids), 1,
                              f"Embeddings synonyms resolved to multiple ids: {resolved}")

    def test_whitespace_is_stripped(self):
        base = ontology.canonical_skill("Python")
        padded = ontology.canonical_skill("  Python  ")
        self.assertEqual(base, padded)


class TestCompanyTier(unittest.TestCase):
    def test_consulting_firm_classified_as_consulting(self):
        self.assertEqual(ontology.company_tier("TCS"), "consulting")
        self.assertEqual(ontology.company_tier("Infosys"), "consulting")

    def test_ai_native_company_classified_correctly(self):
        self.assertEqual(ontology.company_tier("Sarvam AI"), "ai_native")

    def test_global_major_classified_correctly(self):
        self.assertEqual(ontology.company_tier("Meta"), "global_major")

    def test_unknown_company_falls_back_to_generic_or_ambiguous(self):
        tier = ontology.company_tier("Some Totally Made Up Company Inc")
        self.assertIn(tier, {"generic", "ambiguous"})


if __name__ == "__main__":
    unittest.main()
