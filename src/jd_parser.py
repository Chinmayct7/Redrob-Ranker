"""
Rule-based job description parser.

This does NOT do keyword-frequency counting (the trap the JD itself warns
against). It does two things:

1. STRUCTURED EXTRACTION: pulls numeric/categorical constraints out of the
   JD's prose using section-header detection + regex — experience range,
   "ideal" range, notice-period expectations, preferred/welcome locations,
   and the explicit consulting-firm blocklist (parsed straight out of the
   "(TCS, Infosys, Wipro, ... etc.)" parenthetical, not hand-typed).

2. LIVE CROSS-CHECK: the skill-bucket assignments (must-have / nice-to-have
   / watch-out) and the disqualifier rules used by scoring.py are encoded
   as domain knowledge in ontology.py and disqualifiers.py — they were
   *derived* by reading this JD once. At runtime, this parser re-reads the
   actual job_description.md and checks that the key phrases backing each
   of those assumptions are still present. If someone swaps in a
   differently-worded JD before evaluation, this prints visible warnings
   instead of silently scoring against stale assumptions.

Run standalone for a human-readable dump: `python -m src.jd_parser <path>`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class JDRequirements:
    role_title: str
    experience_min: float
    experience_max: float
    ideal_experience_min: float
    ideal_experience_max: float
    notice_ideal_max_days: int
    consulting_firms: set[str]
    preferred_locations: set[str]
    welcome_locations: set[str]
    sections: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown on '#' / '##' headers into {header_text_lower: body}."""
    sections: dict[str, str] = {}
    current_header = "_preamble"
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^#{1,3}\s+\**(.+?)\**\s*$", line.strip())
        if m:
            sections[current_header] = "\n".join(buf).strip()
            current_header = m.group(1).strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[current_header] = "\n".join(buf).strip()
    return sections


def _find_section(sections: dict[str, str], *substrings: str) -> str:
    for header, body in sections.items():
        if all(s in header for s in substrings):
            return body
    return ""


def parse_job_description(text: str) -> JDRequirements:
    sections = _split_sections(text)
    warnings: list[str] = []

    title_match = re.search(r"Job Description:\s*(.+)", text)
    role_title = title_match.group(1).strip() if title_match else "Unknown Role"

    exp_match = re.search(r"Experience Required:\**\s*([\d.]+)\s*[\u2013-]\s*([\d.]+)\s*years", text)
    if exp_match:
        exp_min, exp_max = float(exp_match.group(1)), float(exp_match.group(2))
    else:
        warnings.append("Could not find 'Experience Required: X-Y years' line; defaulting to 5-9.")
        exp_min, exp_max = 5.0, 9.0

    ideal_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years total experience", text)
    if ideal_match:
        ideal_min, ideal_max = float(ideal_match.group(1)), float(ideal_match.group(2))
    else:
        warnings.append("Could not find an explicit 'ideal' years-experience sentence; "
                         "falling back to the midpoint of the required range.")
        mid = (exp_min + exp_max) / 2
        ideal_min, ideal_max = mid - 1, mid + 1

    notice_match = re.search(r"sub-(\d+)-day notice", text)
    notice_ideal = int(notice_match.group(1)) if notice_match else 30
    if not notice_match:
        warnings.append("Could not find explicit notice-period preference; defaulting to 30 days.")

    consulting_section = re.search(
        r"consulting firms[^(]*\(([^)]+)\)", text, flags=re.IGNORECASE
    )
    consulting_firms: set[str] = set()
    if consulting_section:
        raw = consulting_section.group(1)
        for tok in raw.split(","):
            tok = tok.strip().rstrip(".").strip()
            if tok and tok.lower() != "etc":
                consulting_firms.add(tok)
    else:
        warnings.append("Could not find explicit consulting-firm list in JD; "
                         "falling back to the hardcoded list in ontology.py.")

    loc_line = re.search(r"\*\*Location:\*\*\s*(.+)", text)
    preferred_locations: set[str] = set()
    if loc_line:
        for city in re.findall(r"[A-Za-z][A-Za-z .]*", loc_line.group(1).split("|")[0]):
            city = city.strip().lower()
            if city and city not in {"open to relocation candidates from tier", "hybrid", "flexible cadence"}:
                pass  # raw extraction is noisy; primary source of truth below is the welcome-cities sentence
    # Pune/Noida are explicitly the named office locations.
    if "pune" in text.lower():
        preferred_locations.add("pune")
    if "noida" in text.lower():
        preferred_locations.add("noida")

    welcome_match = re.search(r"welcome to apply\.?\s*", text)
    welcome_locations: set[str] = set()
    if welcome_match:
        # Look at the sentence containing "welcome to apply"
        sentence_start = text.rfind(".", 0, welcome_match.start())
        sentence = text[sentence_start + 1: welcome_match.end()]
        for city in re.findall(r"[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?", sentence):
            if city not in {"Outside", "India", "Candidates"}:
                welcome_locations.add(city.lower())
    else:
        warnings.append("Could not find the 'cities welcome to apply' sentence.")

    return JDRequirements(
        role_title=role_title,
        experience_min=exp_min,
        experience_max=exp_max,
        ideal_experience_min=ideal_min,
        ideal_experience_max=ideal_max,
        notice_ideal_max_days=notice_ideal,
        consulting_firms=consulting_firms,
        preferred_locations=preferred_locations,
        welcome_locations=welcome_locations,
        sections=sections,
        warnings=warnings,
    )


def cross_check_assumptions(jd: JDRequirements, full_text: str = "") -> list[str]:
    """Verify the static domain-knowledge tables in ontology.py /
    disqualifiers.py are still grounded in the actual JD text. Returns a
    list of human-readable warnings (empty if everything checks out)."""
    issues: list[str] = []
    need_section = _find_section(jd.sections, "things you absolutely need")
    nice_section = _find_section(jd.sections, "things we'd like")
    dontwant_section = _find_section(jd.sections, "explicitly do not want") or \
        _find_section(jd.sections, "explicitly do n\u2019t want")

    must_have_phrase_hits = {
        "embeddings": "embeddings" in need_section.lower(),
        "vector_db": "vector database" in need_section.lower() or "hybrid search" in need_section.lower(),
        "python": "python" in need_section.lower(),
        "ranking": "ranking" in need_section.lower() or "evaluation framework" in need_section.lower(),
    }
    for skill_id, found in must_have_phrase_hits.items():
        if not found:
            issues.append(f"ontology.py marks '{skill_id}' as must-have, but its trigger "
                           f"phrase was not found in the JD's 'absolutely need' section.")

    disqualifier_phrase_hits = {
        "pure research disqualifier": "pure research" in full_text.lower() or "academic labs" in full_text.lower(),
        "consulting-only disqualifier": "consulting firms" in dontwant_section.lower(),
        "title-chaser disqualifier": "title" in dontwant_section.lower(),
        "cv/speech disqualifier": "computer vision" in dontwant_section.lower() and "speech" in dontwant_section.lower(),
    }
    for label, found in disqualifier_phrase_hits.items():
        if not found:
            issues.append(f"disqualifiers.py implements a '{label}' rule, but its trigger "
                           f"phrase was not found in the JD's disqualifier section.")

    return issues


def load_and_parse(path: str) -> JDRequirements:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    jd = parse_job_description(text)
    jd.warnings.extend(cross_check_assumptions(jd, text))
    return jd


if __name__ == "__main__":
    import sys
    jd = load_and_parse(sys.argv[1] if len(sys.argv) > 1 else "data/job_description.md")
    print(f"Role: {jd.role_title}")
    print(f"Experience required: {jd.experience_min}-{jd.experience_max} (ideal {jd.ideal_experience_min}-{jd.ideal_experience_max})")
    print(f"Notice period ideal: <= {jd.notice_ideal_max_days} days")
    print(f"Consulting firms blocklist (parsed): {sorted(jd.consulting_firms)}")
    print(f"Preferred locations: {jd.preferred_locations}")
    print(f"Welcome locations (parsed, noisy): {jd.welcome_locations}")
    if jd.warnings:
        print("\nWarnings:")
        for w in jd.warnings:
            print(f"  - {w}")
    else:
        print("\nAll static domain-knowledge assumptions cross-checked OK against the live JD text.")
