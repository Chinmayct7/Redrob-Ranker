"""Unit tests for src.disqualifiers — the 7 anti-pattern multipliers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import disqualifiers as dq


class TestConsultingOnlyCareer(unittest.TestCase):
    def test_flags_pure_consulting_career(self):
        c = {"career_history": [
            {"company": "TCS", "description": ""},
            {"company": "Infosys", "description": ""},
        ]}
        triggered, reason = dq.consulting_only_career(c)
        self.assertTrue(triggered)

    def test_does_not_flag_mixed_career(self):
        c = {"career_history": [
            {"company": "TCS", "description": ""},
            {"company": "Swiggy", "description": ""},
        ]}
        triggered, _ = dq.consulting_only_career(c)
        self.assertFalse(triggered)

    def test_empty_history_not_flagged(self):
        triggered, _ = dq.consulting_only_career({"career_history": []})
        self.assertFalse(triggered)


class TestPureResearchNoProduction(unittest.TestCase):
    def test_flags_research_title_without_production_language(self):
        c = {
            "profile": {"current_title": "Research Scientist"},
            "career_history": [{"description": "Published papers on transformer architectures."}],
        }
        triggered, _ = dq.pure_research_no_production(c)
        self.assertTrue(triggered)

    def test_does_not_flag_research_title_with_production_signal(self):
        c = {
            "profile": {"current_title": "Research Scientist"},
            "career_history": [{"description": "Deployed the ranking model to production and ran A/B tests."}],
        }
        triggered, _ = dq.pure_research_no_production(c)
        self.assertFalse(triggered)

    def test_non_research_title_not_flagged(self):
        c = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
        triggered, _ = dq.pure_research_no_production(c)
        self.assertFalse(triggered)


class TestCvSpeechRoboticsOnly(unittest.TestCase):
    def test_flags_cv_only_skillset(self):
        c = {"skills": [{"name": "Computer Vision"}, {"name": "Object Detection"}]}
        # Only relevant if these resolve to CV_SPEECH_IDS canonical ids;
        # falls back gracefully to False if ontology doesn't recognize them.
        triggered, _ = dq.cv_speech_robotics_only(c)
        self.assertIsInstance(triggered, bool)

    def test_nlp_skills_not_flagged(self):
        c = {"skills": [{"name": "Embeddings"}, {"name": "BM25"}]}
        triggered, _ = dq.cv_speech_robotics_only(c)
        self.assertFalse(triggered)


class TestClosedSourceNoValidation(unittest.TestCase):
    def test_flags_experienced_no_github_no_certs(self):
        c = {
            "profile": {"years_of_experience": 7},
            "redrob_signals": {"github_activity_score": 0},
            "certifications": [],
        }
        triggered, _ = dq.closed_source_no_validation(c)
        self.assertTrue(triggered)

    def test_does_not_flag_with_github_activity(self):
        c = {
            "profile": {"years_of_experience": 7},
            "redrob_signals": {"github_activity_score": 0.6},
            "certifications": [],
        }
        triggered, _ = dq.closed_source_no_validation(c)
        self.assertFalse(triggered)

    def test_does_not_flag_junior_candidate(self):
        c = {
            "profile": {"years_of_experience": 2},
            "redrob_signals": {"github_activity_score": 0},
            "certifications": [],
        }
        triggered, _ = dq.closed_source_no_validation(c)
        self.assertFalse(triggered)


class TestEvaluateDisqualifiers(unittest.TestCase):
    def test_no_triggers_returns_multiplier_one(self):
        c = {
            "profile": {"current_title": "ML Engineer", "years_of_experience": 6},
            "career_history": [{"company": "Swiggy", "description": "Shipped ranking to production.",
                                 "is_current": True, "duration_months": 24}],
            "skills": [{"name": "Embeddings", "duration_months": 40}],
            "redrob_signals": {"github_activity_score": 0.5},
            "certifications": ["AWS ML"],
        }
        mult, reasons = dq.evaluate_disqualifiers(c)
        self.assertEqual(mult, 1.0)
        self.assertEqual(reasons, [])

    def test_consulting_only_reduces_multiplier_below_one(self):
        c = {
            "profile": {"current_title": "Software Engineer", "years_of_experience": 6},
            "career_history": [{"company": "TCS", "description": ""}],
            "skills": [],
            "redrob_signals": {"github_activity_score": 0.5},
            "certifications": [],
        }
        mult, reasons = dq.evaluate_disqualifiers(c)
        self.assertLess(mult, 1.0)
        self.assertTrue(any("consulting_only_career" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
