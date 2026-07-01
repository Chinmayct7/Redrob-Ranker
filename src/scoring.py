"""Combines feature components into a single composite score per candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config, disqualifiers, features, semantic


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    components: dict = field(default_factory=dict)
    disqualifier_reasons: list = field(default_factory=list)


def score_candidate(candidate: dict, idf: dict) -> ScoredCandidate:
    text = semantic.candidate_text(candidate)

    f_title = features.title_fit(candidate, text)
    f_skill = features.skill_match(candidate)
    f_exp = features.experience_fit(candidate)
    f_loc = features.location_fit(candidate)
    f_edu = features.education_fit(candidate)
    f_val = features.validation_fit(candidate)
    f_stab = features.career_stability(candidate)
    f_gate = features.signal_gate(candidate)
    semantic_score = semantic.score_candidate_text(text, idf)

    base_fit = (
        config.W_TITLE * f_title.score +
        config.W_SKILL * f_skill.score +
        config.W_SEMANTIC * semantic_score +
        config.W_EXPERIENCE * f_exp.score +
        config.W_LOCATION * f_loc.score +
        config.W_EDUCATION * f_edu.score +
        config.W_VALIDATION * f_val.score +
        config.W_CAREER_STABILITY * f_stab.score
    )

    dq_multiplier, dq_reasons = disqualifiers.evaluate_disqualifiers(candidate)

    final_score = base_fit * dq_multiplier * f_gate.score

    components = {
        "title_fit": f_title, "skill_match": f_skill, "semantic_match": semantic_score,
        "experience_fit": f_exp, "location_fit": f_loc, "education_fit": f_edu,
        "validation_fit": f_val, "career_stability": f_stab, "signal_gate": f_gate,
        "base_fit": base_fit, "dq_multiplier": dq_multiplier,
    }

    return ScoredCandidate(
        candidate_id=candidate["candidate_id"],
        score=final_score,
        components=components,
        disqualifier_reasons=dq_reasons,
    )
