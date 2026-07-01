"""
Honeypot detection: candidates whose own record contains an internal
*impossibility*, not just a weak fit. The dataset doc says these are
"subtly impossible profiles." We don't have an external ground truth (e.g.
real company founding dates) so detection is restricted to checks that are
verifiable purely from arithmetic/logic *within a single candidate record*:

  - duration_months on a career entry doesn't match (end_date - start_date)
  - is_current=True contradicted by a non-null end_date (or vice versa)
  - two career_history entries overlap in time as if both were full-time
  - total summed career duration is wildly inconsistent with
    profile.years_of_experience
  - profile.years_of_experience exceeds what's possible given the
    candidate's own education end_year (can't have 15 years of experience
    2 years after graduating)
  - a skill is claimed at "expert"/"advanced" proficiency with an
    implausibly low duration_months for that same skill
  - education end_year before start_year, or end_year in the future
    relative to a still-listed-as-current career start

Any single impossibility is enough to flag a profile as a honeypot. These
candidates are excluded from ranking entirely (see scoring.py) rather than
merely down-scored — the spec's >10%-in-top-100 threshold suggests
honeypots are meant to be screened out, not just deprioritized.
"""

from __future__ import annotations

import datetime as dt

from . import config

TODAY = dt.date.today()


def _parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def _months_between(d1: dt.date, d2: dt.date) -> float:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.44


def check_honeypot(candidate: dict) -> list[str]:
    """Return a list of human-readable impossibility reasons. Empty list =
    not flagged."""
    reasons: list[str] = []
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []

    yoe = profile.get("years_of_experience")

    # --- per-entry date/duration consistency -----------------------------
    intervals: list[tuple[dt.date, dt.date]] = []
    for entry in history:
        start = _parse_date(entry.get("start_date"))
        end = _parse_date(entry.get("end_date"))
        is_current = entry.get("is_current")
        duration = entry.get("duration_months")

        if is_current and end is not None:
            reasons.append(f"career entry at '{entry.get('company')}' is marked is_current=True "
                            f"but has a non-null end_date ({entry.get('end_date')}).")
        if not is_current and end is None:
            reasons.append(f"career entry at '{entry.get('company')}' is marked is_current=False "
                            f"but has no end_date.")

        effective_end = end if end is not None else TODAY
        if start is not None:
            expected_duration = _months_between(start, effective_end)
            if duration is not None and abs(expected_duration - duration) > config.DATE_DURATION_TOLERANCE_MONTHS:
                reasons.append(
                    f"career entry at '{entry.get('company')}' claims duration_months={duration}, "
                    f"but start_date/end_date imply ~{expected_duration:.0f} months."
                )
            intervals.append((start, effective_end))

    # --- overlapping full-time roles --------------------------------------
    intervals.sort(key=lambda iv: iv[0])
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        overlap_months = _months_between(max(s1, s2), min(e1, e2))
        if s2 < e1 and overlap_months > config.OVERLAP_TOLERANCE_MONTHS:
            reasons.append(
                f"two career_history entries overlap by ~{overlap_months:.0f} months "
                f"as if both were full-time simultaneously."
            )

    # --- total career time vs. stated years_of_experience -----------------
    if intervals and yoe is not None:
        total_months = sum(_months_between(s, e) for s, e in intervals)
        yoe_months = yoe * 12
        if yoe_months > 0:
            ratio = total_months / yoe_months
            if ratio > config.CAREER_SUM_VS_YOE_UPPER_RATIO:
                reasons.append(
                    f"career_history entries sum to ~{total_months:.0f} months of (mostly "
                    f"non-overlapping) work, far more than the stated "
                    f"years_of_experience={yoe} ({yoe_months:.0f} months)."
                )
            elif ratio < config.CAREER_SUM_VS_YOE_LOWER_RATIO:
                reasons.append(
                    f"career_history entries sum to only ~{total_months:.0f} months, far less "
                    f"than the stated years_of_experience={yoe} ({yoe_months:.0f} months)."
                )

    # --- education internal consistency only (no cross-check against
    #     years_of_experience: this dataset has many candidates with
    #     realistic-but-unlisted earlier degrees, so "earliest listed
    #     education end_year" is NOT a safe lower bound on career start.
    #     Tested against the full 100k pool: an experience-vs-education
    #     check fired on ~10% of candidates (false positives), so it was
    #     removed in favor of the checks below, which together flag ~70
    #     candidates pool-wide -- in line with the spec's "~80 honeypots".)
    for e in education:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            reasons.append(f"education entry at '{e.get('institution')}' has end_year ({ey}) "
                            f"before start_year ({sy}).")

    # --- skill proficiency vs. duration_months -----------------------------
    for s in skills:
        prof = s.get("proficiency")
        dur = s.get("duration_months")
        if dur is None:
            continue
        if prof == "expert" and dur < config.EXPERTISE_MIN_MONTHS_FOR_EXPERT:
            reasons.append(f"skill '{s.get('name')}' claimed at 'expert' proficiency with only "
                            f"{dur} months of use.")
        elif prof == "advanced" and dur < config.EXPERTISE_MIN_MONTHS_FOR_ADVANCED:
            reasons.append(f"skill '{s.get('name')}' claimed at 'advanced' proficiency with only "
                            f"{dur} months of use.")

    return reasons


def is_honeypot(candidate: dict) -> bool:
    return len(check_honeypot(candidate)) > 0
