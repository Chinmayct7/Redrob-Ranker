"""Unit tests for src.honeypot — internal-impossibility detection.

Run with: python -m pytest tests/ -v
(or: python -m unittest discover tests)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honeypot import check_honeypot


def _base_candidate(**overrides):
    c = {
        "candidate_id": "CAND_TEST",
        "profile": {
            "years_of_experience": 6.0,
            "current_title": "Machine Learning Engineer",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "ML Engineer",
                "start_date": "2022-06-01",
                "end_date": "2026-06-01",
                "duration_months": 48,
                "is_current": False,
            }
        ],
        "education": [{"institution": "Test University", "start_year": 2014, "end_year": 2018}],
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60}],
    }
    c.update(overrides)
    return c


class TestHoneypotDetection(unittest.TestCase):
    def test_clean_candidate_not_flagged(self):
        reasons = check_honeypot(_base_candidate())
        self.assertEqual(reasons, [])

    def test_is_current_with_end_date_is_flagged(self):
        c = _base_candidate(career_history=[{
            "company": "TestCo", "title": "ML Engineer",
            "start_date": "2020-01-01", "end_date": "2023-01-01",
            "duration_months": 36, "is_current": True,
        }])
        reasons = check_honeypot(c)
        self.assertTrue(any("is_current=True" in r for r in reasons))

    def test_duration_mismatch_is_flagged(self):
        # start->end implies ~12 months but duration_months claims 60
        c = _base_candidate(career_history=[{
            "company": "TestCo", "title": "ML Engineer",
            "start_date": "2022-01-01", "end_date": "2023-01-01",
            "duration_months": 60, "is_current": False,
        }])
        reasons = check_honeypot(c)
        self.assertTrue(any("claims duration_months" in r for r in reasons))

    def test_overlapping_fulltime_roles_flagged(self):
        c = _base_candidate(career_history=[
            {"company": "A", "title": "Eng", "start_date": "2018-01-01",
             "end_date": "2022-01-01", "duration_months": 48, "is_current": False},
            {"company": "B", "title": "Eng", "start_date": "2019-01-01",
             "end_date": "2023-01-01", "duration_months": 48, "is_current": False},
        ])
        reasons = check_honeypot(c)
        self.assertTrue(any("overlap" in r for r in reasons))

    def test_career_sum_far_exceeds_stated_yoe(self):
        c = _base_candidate(
            profile={"years_of_experience": 2.0, "current_title": "ML Engineer"},
            career_history=[{
                "company": "A", "title": "Eng", "start_date": "2010-01-01",
                "end_date": "2022-01-01", "duration_months": 144, "is_current": False,
            }],
        )
        reasons = check_honeypot(c)
        self.assertTrue(any("far more than the stated" in r for r in reasons))

    def test_expert_skill_with_low_duration_flagged(self):
        c = _base_candidate(skills=[
            {"name": "Embeddings", "proficiency": "expert", "duration_months": 1}
        ])
        reasons = check_honeypot(c)
        self.assertTrue(any("expert" in r and "Embeddings" in r for r in reasons))

    def test_education_end_before_start_flagged(self):
        c = _base_candidate(education=[
            {"institution": "X", "start_year": 2020, "end_year": 2016}
        ])
        reasons = check_honeypot(c)
        self.assertTrue(any("end_year" in r and "before start_year" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
