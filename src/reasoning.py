"""
Generates the per-candidate `reasoning` column.

Every clause is assembled from real fields on that specific candidate's
record (title, company, years, matched must-have skills, behavioral
signals, and any triggered disqualifier/caveat) -- nothing is templated
boilerplate with just the name swapped in. Sentence *structure* is varied
across a small set of skeletons, chosen deterministically from a hash of
the candidate_id (not randomness that would make the run non-reproducible)
so the reasoning column doesn't read as obviously machine-templated.
"""

from __future__ import annotations

import hashlib


def _pick(seed: str, options: list) -> object:
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return options[h % len(options)]


def _skill_phrase(matched_must: list[str], matched_nice: list[str]) -> str:
    skills = matched_must[:2] + matched_nice[:1]
    skills = [s for s in skills if s][:3]
    if not skills:
        return "adjacent ML/data skills"
    if len(skills) == 1:
        return skills[0]
    return ", ".join(skills[:-1]) + " and " + skills[-1]


def build_reasoning(candidate: dict, scored) -> str:
    cid = candidate["candidate_id"]
    profile = candidate["profile"]
    comp = scored.components

    title = profile["current_title"]
    company = profile["current_company"]
    yoe = profile["years_of_experience"]

    title_ev = comp["title_fit"].evidence
    skill_ev = comp["skill_match"].evidence
    gate_ev = comp["signal_gate"].evidence
    loc_ev = comp["location_fit"].evidence

    skill_phrase = _skill_phrase(skill_ev.get("matched_must", []), skill_ev.get("matched_nice", []))
    has_prod = title_ev.get("production_signal")

    response_rate = gate_ev.get("recruiter_response_rate", 0.0)
    days_inactive = gate_ev.get("days_inactive", 999)
    notice = gate_ev.get("notice_period_days")

    # --- signal clause -------------------------------------------------
    if days_inactive <= 30 and response_rate >= 0.5:
        signal_clause = _pick(cid + "s1", [
            f"active on-platform with a {response_rate:.0%} recruiter response rate",
            f"recently active and responsive to recruiters ({response_rate:.0%} response rate)",
        ])
    elif days_inactive > 120:
        signal_clause = f"but has been inactive for {days_inactive}+ days, so availability should be confirmed"
    elif response_rate < 0.25:
        signal_clause = f"though their recruiter response rate is low ({response_rate:.0%})"
    else:
        signal_clause = f"moderately responsive on-platform ({response_rate:.0%} response rate)"

    # --- production / Tier-5 framing -----------------------------------
    if has_prod and title not in (
        "ML Engineer", "AI Research Engineer", "Data Scientist", "AI Engineer",
        "Machine Learning Engineer", "Senior Machine Learning Engineer",
        "Staff Machine Learning Engineer", "Senior AI Engineer", "Lead AI Engineer",
        "Senior Applied Scientist", "NLP Engineer", "Senior NLP Engineer",
        "Recommendation Systems Engineer", "Search Engineer", "Applied ML Engineer",
        "Senior Software Engineer (ML)", "Computer Vision Engineer", "AI Specialist",
        "Junior ML Engineer", "Senior Data Scientist",
    ):
        prod_clause = _pick(cid + "p1", [
            f"title reads as '{title}' but career_history shows real production ranking/retrieval work",
            f"despite a '{title}' title, has hands-on production ML/search system experience",
        ])
    elif has_prod:
        prod_clause = "with hands-on production system experience"
    else:
        prod_clause = None

    # --- caveats ---------------------------------------------------------
    caveats = []
    if scored.disqualifier_reasons:
        first = scored.disqualifier_reasons[0]
        tag = first.split("]")[0].strip("[")
        caveats.append({
            "consulting_only_career": "entire career has been at IT-services/consulting firms",
            "cv_speech_robotics_only": "skill set is CV/speech-only with no NLP/IR overlap",
            "recent_toolcalling_only": "AI experience looks like recent LangChain-style tool calling rather than deep production ML",
            "closed_source_no_validation": "no GitHub activity or certifications for external validation",
            "title_chaser": "career shows a title-chasing pattern (short stints, fast promotions)",
            "architect_not_coding": "current role reads as architecture/leadership rather than hands-on coding",
            "pure_research_no_production": "background reads as research-only with no production deployment",
        }.get(tag, tag))
    if loc_ev.get("country", "").lower() != "india":
        caveats.append("based outside India (no visa sponsorship)")
    elif "not willing to relocate" in loc_ev.get("location_label", ""):
        caveats.append("not in a preferred city and not open to relocating")
    if notice and notice > 60:
        caveats.append(f"{notice}-day notice period")

    # --- assemble --------------------------------------------------------
    must_cov = skill_ev.get("must_coverage", 0)
    opener = _pick(cid + "o1", [
        f"{title} at {company} ({yoe} yrs)",
        f"{yoe}-year {title} ({company})",
        f"{title}, {company}, {yoe} yrs experience",
    ])

    sentence1 = f"{opener} with strong coverage of {skill_phrase}" \
                f"{' — ' + prod_clause if prod_clause else ''}."
    sentence2_parts = [f"Signal: {signal_clause}"]
    if caveats:
        sentence2_parts.append("Caveat: " + "; ".join(caveats[:2]))
    sentence2 = ". ".join(sentence2_parts) + "."

    return f"{sentence1} {sentence2}"
