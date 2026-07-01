"""
Per-candidate feature scoring. Every function returns a small dataclass
with a 0..1 `score` plus `evidence` -- concrete, grounded facts the
reasoning generator can quote (so reasoning text is assembled from real
matched fields, not templated boilerplate with the name swapped in).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import config, ontology

TODAY = dt.date.today()


@dataclass
class FeatureResult:
    score: float
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Title / role fit
# ---------------------------------------------------------------------

def _has_production_signal(text: str) -> bool:
    return any(p in text for p in ontology.PRODUCTION_SYSTEM_PHRASES)


def title_fit(candidate: dict, full_text: str) -> FeatureResult:
    title = candidate["profile"]["current_title"]
    company = candidate["profile"]["current_company"]
    prod_signal = _has_production_signal(full_text)

    if title in ontology.GENERIC_NONTECH_TITLES:
        base = 0.05
        bonus = 0.40 if prod_signal else 0.0
        tier_label = "non-technical"
    elif title in ontology.GENERAL_TECH_TITLES:
        base = 0.32
        bonus = 0.35 if prod_signal else 0.0
        tier_label = "general software engineering"
    elif title in ontology.DATA_ADJACENT_TITLES:
        base = 0.52
        bonus = 0.35 if prod_signal else 0.0
        tier_label = "data/backend engineering (Tier-5 candidate pool)"
    else:
        # Elimination default: anything not in the three buckets above is
        # an AI/ML-specific title (closed, known vocabulary -- see
        # ontology.py docstring for the full enumeration check).
        base = 0.78
        bonus = 0.18 if prod_signal else 0.0
        tier_label = "AI/ML-specific"

    tier = ontology.company_tier(company)
    company_bonus = {"ai_native": 0.10, "product": 0.05, "global_major": 0.05,
                      "consulting": -0.05, "ambiguous": 0.0, "generic": 0.0}[tier]

    score = max(0.0, min(1.0, base + bonus + company_bonus))
    return FeatureResult(score, {
        "title": title, "title_tier": tier_label, "production_signal": prod_signal,
        "company": company, "company_tier": tier,
    })


# ---------------------------------------------------------------------
# Skill match (ontology-aware, trust-adjusted against assessment scores)
# ---------------------------------------------------------------------

def skill_match(candidate: dict) -> FeatureResult:
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    per_canonical: dict[str, float] = {}
    matched_labels: dict[str, list[str]] = {"must": [], "nice": [], "adjacent": [], "watch": []}

    for s in candidate.get("skills", []) or []:
        name = s.get("name", "")
        cid = ontology.canonical_skill(name)
        if cid is None:
            continue
        bucket_info = ontology.CANONICAL_SKILLS[cid]
        bucket = bucket_info["bucket"]

        prof_weight = ontology.PROFICIENCY_WEIGHT.get(s.get("proficiency"), 0.4)
        duration = s.get("duration_months") or 0
        recency = max(0.15, min(1.0, duration / 24))

        trust = 1.0
        if name in assessments:
            implied = prof_weight
            assessed = assessments[name] / 100.0
            if assessed < implied - 0.20:
                trust = max(0.30, assessed / implied if implied > 0 else 0.30)
            elif assessed > implied:
                trust = min(1.15, 1.0 + (assessed - implied) * 0.3)

        endorsements = s.get("endorsements") or 0
        endorse_factor = 1.0 + min(0.15, (endorsements / 100.0) * 0.15)

        contribution = prof_weight * recency * trust * endorse_factor
        if cid not in per_canonical or contribution > per_canonical[cid]:
            per_canonical[cid] = contribution
            matched_labels[bucket] = [l for l in matched_labels[bucket] if not l.startswith(bucket_info["label"])]
        if contribution >= 0.5 and bucket_info["label"] not in matched_labels[bucket]:
            matched_labels[bucket].append(bucket_info["label"])

    def coverage(id_set):
        if not id_set:
            return 0.0
        return sum(min(1.0, per_canonical.get(cid, 0.0)) for cid in id_set) / len(id_set)

    must_cov = coverage(ontology.MUST_HAVE_IDS)
    nice_cov = coverage(ontology.NICE_TO_HAVE_IDS)
    adj_cov = coverage(ontology.ADJACENT_IDS)
    watch_cov = coverage(ontology.WATCH_IDS)

    score = max(0.0, min(1.0, 0.62 * must_cov + 0.25 * nice_cov + 0.08 * adj_cov + 0.05 * watch_cov))
    return FeatureResult(score, {
        "must_coverage": round(must_cov, 2), "nice_coverage": round(nice_cov, 2),
        "matched_must": matched_labels["must"], "matched_nice": matched_labels["nice"],
        "matched_adjacent": matched_labels["adjacent"], "matched_watch": matched_labels["watch"],
    })


# ---------------------------------------------------------------------
# Experience years fit
# ---------------------------------------------------------------------

def experience_fit(candidate: dict) -> FeatureResult:
    yoe = candidate["profile"].get("years_of_experience", 0)
    if config.EXP_IDEAL_MIN <= yoe <= config.EXP_IDEAL_MAX:
        score = 1.0
    elif config.EXP_ACCEPTABLE_MIN <= yoe < config.EXP_IDEAL_MIN:
        score = 0.75 + 0.25 * (yoe - config.EXP_ACCEPTABLE_MIN) / (config.EXP_IDEAL_MIN - config.EXP_ACCEPTABLE_MIN)
    elif config.EXP_IDEAL_MAX < yoe <= config.EXP_ACCEPTABLE_MAX:
        score = 1.0 - 0.25 * (yoe - config.EXP_IDEAL_MAX) / (config.EXP_ACCEPTABLE_MAX - config.EXP_IDEAL_MAX)
    elif yoe < config.EXP_ACCEPTABLE_MIN:
        span = max(config.EXP_ACCEPTABLE_MIN - config.EXP_HARD_FLOOR, 0.1)
        score = max(0.0, 0.75 * (yoe - config.EXP_HARD_FLOOR) / span)
    else:  # yoe > acceptable max
        over = yoe - config.EXP_ACCEPTABLE_MAX
        score = max(0.15, 0.75 - 0.08 * over)
    return FeatureResult(max(0.0, min(1.0, score)), {"years_of_experience": yoe})


# ---------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------

def location_fit(candidate: dict) -> FeatureResult:
    profile = candidate["profile"]
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country != "india":
        score = 0.30 if relocate else 0.15
        label = "outside India (JD: case-by-case, no visa sponsorship)"
    elif any(c in location for c in config.PREFERRED_LOCATIONS):
        score = 1.0
        label = "in a preferred office city (Pune/Noida)"
    elif any(c in location for c in config.WELCOME_LOCATIONS):
        score = 0.85
        label = "in an explicitly welcomed Tier-1 city"
    elif relocate:
        score = 0.55
        label = "elsewhere in India, but willing to relocate"
    else:
        score = 0.30
        label = "elsewhere in India, not willing to relocate"

    return FeatureResult(score, {"location": profile.get("location"), "country": profile.get("country"),
                                  "willing_to_relocate": relocate, "location_label": label})


# ---------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------

RELEVANT_FIELDS = {"computer science", "data science", "artificial intelligence",
                    "machine learning", "statistics", "mathematics", "information technology"}
TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.45}


def education_fit(candidate: dict) -> FeatureResult:
    education = candidate.get("education", []) or []
    if not education:
        return FeatureResult(0.4, {"note": "no education listed"})
    best = max(education, key=lambda e: TIER_SCORE.get(e.get("tier", "unknown"), 0.45))
    tier_score = TIER_SCORE.get(best.get("tier", "unknown"), 0.45)
    field_bonus = 0.15 if (best.get("field_of_study") or "").lower() in RELEVANT_FIELDS else 0.0
    score = max(0.0, min(1.0, tier_score + field_bonus))
    return FeatureResult(score, {"institution": best.get("institution"), "degree": best.get("degree"),
                                  "field_of_study": best.get("field_of_study"), "tier": best.get("tier")})


# ---------------------------------------------------------------------
# External validation (GitHub / certifications) -- a positive-only bonus,
# distinct from the mild closed_source_no_validation disqualifier penalty.
# ---------------------------------------------------------------------

def validation_fit(candidate: dict) -> FeatureResult:
    gh = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    certs = candidate.get("certifications", []) or []
    gh_score = 0.0 if gh < 0 else min(1.0, gh / 70.0)
    cert_score = min(1.0, len(certs) * 0.25)
    score = max(0.0, min(1.0, 0.75 * gh_score + 0.25 * cert_score))
    return FeatureResult(score, {"github_activity_score": gh, "num_certifications": len(certs)})


# ---------------------------------------------------------------------
# Career stability (distinct from the title_chaser hard disqualifier --
# this is a smoother reward for sane average tenure)
# ---------------------------------------------------------------------

def career_stability(candidate: dict) -> FeatureResult:
    history = candidate.get("career_history", []) or []
    durations = [h.get("duration_months") or 0 for h in history if h.get("duration_months")]
    if not durations:
        return FeatureResult(0.5, {"avg_tenure_months": None})
    avg = sum(durations) / len(durations)
    # Sweet spot ~18-48 months average tenure; too short (hopping) or
    # absurdly long-only (one job forever, no breadth) both score a bit
    # lower, but the penalty for long tenure is much gentler.
    if 18 <= avg <= 48:
        score = 1.0
    elif avg < 18:
        score = max(0.2, avg / 18)
    else:
        score = max(0.6, 1.0 - (avg - 48) / 200)
    return FeatureResult(score, {"avg_tenure_months": round(avg, 1), "num_roles": len(history)})


# ---------------------------------------------------------------------
# Behavioral signal quality / availability gate (multiplicative, applied
# on top of the base fit score -- a great skill match attached to a
# 6-months-dark profile should rank below an equally good match who is
# actually reachable, per the JD's closing note.)
# ---------------------------------------------------------------------

def signal_gate(candidate: dict) -> FeatureResult:
    sig = candidate.get("redrob_signals", {})

    last_active = sig.get("last_active_date")
    try:
        days_inactive = (TODAY - dt.date.fromisoformat(last_active)).days
    except (TypeError, ValueError):
        days_inactive = config.LAST_ACTIVE_ZERO_CREDIT_DAYS
    if days_inactive <= config.LAST_ACTIVE_FULL_CREDIT_DAYS:
        recency = 1.0
    elif days_inactive >= config.LAST_ACTIVE_ZERO_CREDIT_DAYS:
        recency = 0.15
    else:
        span = config.LAST_ACTIVE_ZERO_CREDIT_DAYS - config.LAST_ACTIVE_FULL_CREDIT_DAYS
        recency = 1.0 - 0.85 * (days_inactive - config.LAST_ACTIVE_FULL_CREDIT_DAYS) / span

    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.4
    response_rate = sig.get("recruiter_response_rate", 0.5) or 0.0
    interview_completion = sig.get("interview_completion_rate", 0.5) or 0.0

    oar = sig.get("offer_acceptance_rate", -1)
    offer_acceptance = 0.7 if oar is None or oar < 0 else oar

    verification = sum([
        1.0 if sig.get("verified_email") else 0.0,
        1.0 if sig.get("verified_phone") else 0.0,
        1.0 if sig.get("linkedin_connected") else 0.0,
    ]) / 3.0

    notice = sig.get("notice_period_days", 30) or 0
    if notice <= config.NOTICE_IDEAL_MAX_DAYS:
        notice_score = 1.0
    elif notice >= config.NOTICE_HARD_MAX_DAYS:
        notice_score = 0.3
    else:
        span = config.NOTICE_HARD_MAX_DAYS - config.NOTICE_IDEAL_MAX_DAYS
        notice_score = 1.0 - 0.7 * (notice - config.NOTICE_IDEAL_MAX_DAYS) / span

    completeness = (sig.get("profile_completeness_score", 70) or 0) / 100.0

    # --- Signal 8: avg_response_time_hours ---
    # Faster responders are more hireable. Cap at 72h for scoring.
    avg_resp_hours = sig.get("avg_response_time_hours", 24) or 24
    response_speed = max(0.0, 1.0 - (avg_resp_hours / 72.0))

    # --- Signal 17: search_appearance_30d ---
    # How often recruiters are actively finding this profile.
    # Normalize: 0 appearances = 0.0, 20+ = 1.0
    search_appearance = min(1.0, (sig.get("search_appearance_30d", 0) or 0) / 20.0)

    # --- Signal 18: saved_by_recruiters_30d ---
    # Strong hiring-intent signal: recruiter saved = active interest.
    # Normalize: 0 saves = 0.3 (neutral), 5+ = 1.0
    saves = sig.get("saved_by_recruiters_30d", 0) or 0
    saved_score = 0.3 + min(0.7, saves / 5.0)

    # --- Signal 6: applications_submitted_30d ---
    # Shows candidate is actively job-hunting (positive), but 0 is
    # neutral (not penalized — they might be passively open).
    apps = sig.get("applications_submitted_30d", 0) or 0
    application_activity = 0.5 + min(0.5, apps / 10.0)  # 0.5 neutral, max 1.0

    # --- Signal 10: connection_count ---
    # Proxy for professional network / platform engagement.
    # Normalize: 0 = 0.3, 200+ = 1.0
    connections = sig.get("connection_count", 0) or 0
    network_score = 0.3 + min(0.7, connections / 200.0)

    # --- Signal 11: endorsements_received (total, not per-skill) ---
    # Overall social proof on platform.
    endorsements = sig.get("endorsements_received", 0) or 0
    endorsement_score = min(1.0, endorsements / 50.0)

    # --- Signal 14: preferred_work_mode ---
    # JD is hybrid (Pune/Noida offices). Flexible/hybrid = best fit.
    work_mode = (sig.get("preferred_work_mode") or "flexible").lower()
    work_mode_score = {"flexible": 1.0, "hybrid": 1.0, "onsite": 0.8, "remote": 0.5}.get(work_mode, 0.7)

    # --- Signal 13: expected_salary_range_inr_lpa ---
    # No explicit budget in JD, so just check for extreme outliers.
    # We use this as a mild signal only (small weight).
    sal = sig.get("expected_salary_range_inr_lpa", {}) or {}
    sal_min = sal.get("min", 0) or 0
    sal_max = sal.get("max", 999) or 999
    # Flag unrealistic asks (>80 LPA max for a 5-9 yr role) as mild negative
    salary_fit = 0.7 if sal_max > 80 else 1.0

    # --- Signal 2: signup_date ---
    # Longer tenure on platform = more behavioral signal data available.
    signup = sig.get("signup_date")
    try:
        days_on_platform = (TODAY - dt.date.fromisoformat(signup)).days
        platform_tenure = min(1.0, days_on_platform / 365.0)
    except (TypeError, ValueError):
        platform_tenure = 0.5

    weights = {
        # Core availability signals (highest weights)
        "recency":             (recency,             0.20),
        "response_rate":       (response_rate,        0.14),
        "open_to_work":        (open_to_work,         0.07),
        "interview_completion":(interview_completion, 0.07),
        "offer_acceptance":    (offer_acceptance,     0.07),
        # Responsiveness
        "response_speed":      (response_speed,       0.06),
        # Platform engagement
        "saved_score":         (saved_score,          0.06),
        "search_appearance":   (search_appearance,    0.05),
        "application_activity":(application_activity, 0.04),
        "network_score":       (network_score,        0.03),
        "endorsement_score":   (endorsement_score,    0.03),
        # Profile quality
        "verification":        (verification,         0.07),
        "completeness":        (completeness,         0.05),
        "platform_tenure":     (platform_tenure,      0.02),
        # Logistics
        "notice":              (notice_score,         0.07),
        "work_mode_score":     (work_mode_score,      0.04),
        "salary_fit":          (salary_fit,           0.03),
    }
    weighted_avg = sum(v * w for v, w in weights.values()) / sum(w for _, w in weights.values())
    gate = config.SIGNAL_GATE_MIN + weighted_avg * (config.SIGNAL_GATE_MAX - config.SIGNAL_GATE_MIN)

    return FeatureResult(gate, {
        "days_inactive": days_inactive, "last_active_date": last_active,
        "open_to_work_flag": sig.get("open_to_work_flag"),
        "recruiter_response_rate": response_rate,
        "avg_response_time_hours": avg_resp_hours,
        "notice_period_days": notice,
        "offer_acceptance_rate": oar,
        "saved_by_recruiters_30d": saves,
        "search_appearance_30d": sig.get("search_appearance_30d", 0),
        "preferred_work_mode": work_mode,
        "expected_salary_max_lpa": sal_max,
    })
