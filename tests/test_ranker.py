"""
Smoke tests for the Redrob ranker. Run with:

    python -m pytest tests/ -v

or, dependency-free:

    python tests/test_ranker.py

These are not exhaustive — they exist to catch the failure modes that
would sink a submission outright: a honeypot slipping through, a
disqualifier no longer firing, or rank.py silently breaking on the
official sample data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.honeypot import check_honeypot
from src.disqualifiers import evaluate_disqualifiers
from src import scoring, semantic, io_utils
from src.jd_parser import load_and_parse


def _load_sample():
    with open(ROOT / "data" / "sample_candidates.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# honeypot.py
# ---------------------------------------------------------------------------

def test_honeypot_catches_duration_mismatch():
    bad = {
        "candidate_id": "TEST_1",
        "profile": {"years_of_experience": 5},
        "career_history": [{
            "company": "Acme", "title": "Engineer",
            "start_date": "2020-01-01", "end_date": "2020-02-01",
            "duration_months": 48, "is_current": False,
        }],
        "skills": [],
    }
    assert check_honeypot(bad), "duration_months wildly inconsistent with dates should be flagged"


def test_honeypot_catches_impossible_expertise():
    bad = {
        "candidate_id": "TEST_2",
        "profile": {"years_of_experience": 3},
        "career_history": [],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 1}],
    }
    assert check_honeypot(bad), "expert proficiency with 1 month of use should be flagged"


def test_honeypot_leaves_clean_profile_alone():
    clean = {
        "candidate_id": "TEST_3",
        "profile": {"years_of_experience": 6},
        "career_history": [{
            "company": "Acme", "title": "ML Engineer",
            "start_date": "2020-01-01", "end_date": "2023-01-01",
            "duration_months": 36, "is_current": False,
        }],
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60}],
    }
    assert not check_honeypot(clean), "internally-consistent profile should not be flagged"


def test_honeypot_rate_on_sample_is_low():
    """Sanity bound: honeypots should be a small minority, never the majority."""
    sample = _load_sample()
    flagged = sum(1 for c in sample if check_honeypot(c))
    assert flagged / len(sample) < 0.25


# ---------------------------------------------------------------------------
# disqualifiers.py
# ---------------------------------------------------------------------------

def test_consulting_only_career_triggers():
    candidate = {
        "profile": {"current_title": "Software Engineer"},
        "career_history": [
            {"company": "Tata Consultancy Services", "title": "Engineer"},
            {"company": "Infosys", "title": "Senior Engineer"},
        ],
        "skills": [],
        "redrob_signals": {},
    }
    mult, reasons = evaluate_disqualifiers(candidate)
    assert mult < 1.0
    assert any("consulting_only_career" in r for r in reasons)


def test_no_disqualifiers_for_clean_candidate():
    candidate = {
        "profile": {"current_title": "Senior ML Engineer"},
        "career_history": [{"company": "Razorpay", "title": "ML Engineer",
                             "description": "Built and deployed a ranking model at scale.",
                             "is_current": True, "duration_months": 30}],
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60}],
        "redrob_signals": {"github_activity_score": 0.6},
        "certifications": [],
    }
    mult, reasons = evaluate_disqualifiers(candidate)
    assert mult == 1.0
    assert reasons == []


# ---------------------------------------------------------------------------
# end-to-end on the official sample bundle
# ---------------------------------------------------------------------------

def test_end_to_end_on_sample_candidates():
    jd = load_and_parse(str(ROOT / "data" / "job_description.md"))
    sample = _load_sample()

    idf, n_seen = semantic.compute_idf(iter(semantic.candidate_text(c) for c in sample))
    assert n_seen == len(sample)

    scored_any = False
    for c in sample:
        if check_honeypot(c):
            continue
        result = scoring.score_candidate(c, idf)
        assert 0.0 <= result.score <= 1.5  # gate can push slightly above 1.0 in principle
        scored_any = True
    assert scored_any, "every sample candidate was excluded as a honeypot -- suspicious"


if __name__ == "__main__":
    # Dependency-free runner: call every test_* function and report failures.
    import traceback

    tests = [(name, fn) for name, fn in list(globals().items())
              if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception:
            failures += 1
            print(f"  ERROR {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
