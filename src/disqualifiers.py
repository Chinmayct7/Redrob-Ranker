"""
Anti-pattern detectors corresponding to job_description.md's explicit
"things we explicitly do NOT want" and "disqualifiers we actually apply"
sections. Each returns (triggered: bool, reason: str | None). Each maps to
a multiplicative penalty in config.py rather than a hard exclusion — only
honeypot.py produces a hard exclusion — because the JD itself hedges most
of these with "probably" rather than stating an absolute.

Every check below was tested against the full 100k-candidate pool before
being finalized (see scripts/calibrate_disqualifiers.py) specifically to
catch checks that fire on an implausibly large fraction of candidates,
the way an early version of the honeypot education-timeline check did.
"""

from __future__ import annotations

from . import config, ontology


def _all_descriptions(candidate: dict) -> str:
    parts = [candidate.get("profile", {}).get("summary", "")]
    parts += [h.get("description", "") for h in candidate.get("career_history", [])]
    return " ".join(parts).lower()


def _has_production_signal(candidate: dict) -> bool:
    text = _all_descriptions(candidate)
    return any(phrase in text for phrase in ontology.PRODUCTION_SYSTEM_PHRASES)


def consulting_only_career(candidate: dict) -> tuple[bool, str | None]:
    history = candidate.get("career_history", [])
    if not history:
        return False, None
    companies = {h.get("company") for h in history}
    if companies and companies.issubset(ontology.CONSULTING_SERVICES_FIRMS):
        return True, ("entire career_history is at IT-services/consulting firms "
                       f"({', '.join(sorted(companies))}) with no product-company experience.")
    return False, None


def pure_research_no_production(candidate: dict) -> tuple[bool, str | None]:
    title = candidate.get("profile", {}).get("current_title", "").lower()
    is_research_title = any(tok in title for tok in ontology.RESEARCH_ONLY_TITLE_TOKENS)
    if not is_research_title:
        return False, None
    if _has_production_signal(candidate):
        return False, None
    return True, f"title '{candidate['profile']['current_title']}' reads as research-only, " \
                 f"with no production/deployment language anywhere in career_history."


def architect_not_coding(candidate: dict) -> tuple[bool, str | None]:
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if not any(tok in title for tok in ontology.ARCHITECT_NONCODING_TITLE_TOKENS):
        return False, None
    current = next((h for h in candidate.get("career_history", []) if h.get("is_current")), None)
    if current and (current.get("duration_months") or 0) > config.TITLE_CHASER_MAX_TENURE_MONTHS:
        return True, (f"current title '{candidate['profile']['current_title']}' suggests an "
                       f"architecture/leadership role held for {current.get('duration_months')} "
                       f"months, away from hands-on coding.")
    return False, None


def title_chaser(candidate: dict) -> tuple[bool, str | None]:
    history = sorted(
        (h for h in candidate.get("career_history", []) if h.get("start_date")),
        key=lambda h: h["start_date"],
    )
    ranked = []
    for h in history:
        title = h.get("title", "").lower()
        rank = max((i for i, tok in enumerate(ontology.TITLE_LADDER) if tok in title), default=None)
        if rank is not None:
            ranked.append((rank, h.get("duration_months") or 0, h.get("company")))

    jumps = 0
    for (r1, d1, c1), (r2, d2, c2) in zip(ranked, ranked[1:]):
        if r2 > r1 and d1 <= config.TITLE_CHASER_MAX_TENURE_MONTHS:
            jumps += 1
    if jumps >= config.TITLE_CHASER_MIN_JUMPS:
        return True, (f"career_history shows {jumps} title-ladder promotions each preceded by "
                       f"a stint of <= {config.TITLE_CHASER_MAX_TENURE_MONTHS} months "
                       f"(title-chasing pattern).")
    return False, None


def cv_speech_robotics_only(candidate: dict) -> tuple[bool, str | None]:
    skill_ids = {ontology.canonical_skill(s["name"]) for s in candidate.get("skills", [])}
    skill_ids.discard(None)
    has_cv_speech = bool(skill_ids & ontology.CV_SPEECH_IDS)
    has_nlp_ir = bool(skill_ids & ontology.NLP_IR_IDS)
    if has_cv_speech and not has_nlp_ir:
        return True, "skill set is computer-vision/speech-only with no NLP/IR overlap."
    return False, None


def recent_toolcalling_only(candidate: dict) -> tuple[bool, str | None]:
    skills = candidate.get("skills", [])
    ai_skills = [(s, ontology.canonical_skill(s["name"])) for s in skills]
    ai_skills = [(s, cid) for s, cid in ai_skills if cid is not None]
    if not ai_skills:
        return False, None

    has_tooling = any(cid == "llm_tooling" for _, cid in ai_skills)
    if not has_tooling:
        return False, None

    max_ai_duration = max((s.get("duration_months") or 0) for s, _ in ai_skills)
    deep_prelm_ids = {"ml_general", "deep_learning", "statistical_modeling", "ml_framework", "data_science"}
    has_deep_prelm = any(
        cid in deep_prelm_ids and (s.get("duration_months") or 0) >= 24
        for s, cid in ai_skills
    )
    if max_ai_duration < 12 and not has_deep_prelm:
        return True, ("all AI-tagged skills have under 12 months of use and include LLM-tooling "
                       "(LangChain/LlamaIndex/Haystack-style) skills, with no substantial "
                       "pre-LLM-era ML production experience to offset it.")
    return False, None


def closed_source_no_validation(candidate: dict) -> tuple[bool, str | None]:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    gh = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    certs = candidate.get("certifications", [])
    if yoe >= 5 and gh <= 0 and not certs:
        return True, (f"{yoe} years of experience with no GitHub activity "
                       f"(github_activity_score={gh}) and no certifications as external validation.")
    return False, None


ALL_CHECKS = [
    ("consulting_only_career", consulting_only_career, config.MULT_CONSULTING_ONLY_CAREER),
    ("pure_research_no_production", pure_research_no_production, config.MULT_PURE_RESEARCH_NO_PRODUCTION),
    ("architect_not_coding", architect_not_coding, config.MULT_ARCHITECT_NOT_CODING),
    ("title_chaser", title_chaser, config.MULT_TITLE_CHASER),
    ("cv_speech_robotics_only", cv_speech_robotics_only, config.MULT_CV_SPEECH_ROBOTICS_ONLY),
    ("recent_toolcalling_only", recent_toolcalling_only, config.MULT_RECENT_TOOLCALLING_ONLY),
    ("closed_source_no_validation", closed_source_no_validation, config.MULT_CLOSED_SOURCE_NO_VALIDATION),
]


def evaluate_disqualifiers(candidate: dict) -> tuple[float, list[str]]:
    """Returns (combined_multiplier, list_of_triggered_reasons)."""
    multiplier = 1.0
    reasons = []
    for name, fn, mult in ALL_CHECKS:
        triggered, reason = fn(candidate)
        if triggered:
            multiplier *= mult
            reasons.append(f"[{name}] {reason}")
    return multiplier, reasons
