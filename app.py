import csv
import gzip
import io
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import config, io_utils, scoring, semantic
from src.honeypot import check_honeypot
from src.jd_parser import load_and_parse
from src.reasoning import build_reasoning

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .score-bar-bg {
    background: #f0f0f0; border-radius: 4px; height: 8px; width: 100%; margin-top: 4px;
  }
  .score-bar-fill {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
  }

  .rank-badge {
    display: inline-block; background: #1e293b; color: #f8fafc;
    border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 500;
  }
  .score-badge {
    display: inline-block; background: #dbeafe; color: #1d4ed8;
    border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
  }
  .tag {
    display: inline-block; background: #f1f5f9; color: #475569;
    border-radius: 4px; padding: 2px 8px; font-size: 0.72rem;
    margin: 2px; font-weight: 500;
  }
  .tag-must {
    background: #dcfce7; color: #166534;
  }
  .tag-nice {
    background: #dbeafe; color: #1d4ed8;
  }
  .tag-warn {
    background: #fef9c3; color: #854d0e;
  }
  .tag-bad {
    background: #fee2e2; color: #991b1b;
  }
  .caveat-box {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0; padding: 8px 12px;
    font-size: 0.82rem; color: #78350f; margin-top: 6px;
  }
  .metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .metric-number {
    font-size: 2rem; font-weight: 700; color: #1e293b;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label {
    font-size: 0.75rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 2px;
  }

  div[data-testid="stExpander"] summary {
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)




def score_bar(value: float, color="#2563eb") -> str:
    pct = int(value * 100)
    return f"""
    <div class='score-bar-bg'>
      <div class='score-bar-fill' style='width:{pct}%; background:{color};'></div>
    </div>
    """


def component_color(score: float) -> str:
    if score >= 0.75: return "#16a34a"
    if score >= 0.5:  return "#2563eb"
    if score >= 0.25: return "#d97706"
    return "#dc2626"


def load_candidates_from_upload(uploaded_file) -> list[dict]:
    raw = uploaded_file.read()
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except Exception:
        text = raw.decode("utf-8")
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return candidates


@st.cache_data(show_spinner=False)
def run_ranking(candidates_json_str: str, jd_path: str, top_n: int):
    """Cached ranking so re-renders don't re-score."""
    candidates = [json.loads(l) for l in candidates_json_str.strip().splitlines() if l.strip()]
    jd = load_and_parse(jd_path)


    texts = [semantic.candidate_text(c) for c in candidates if not check_honeypot(c)]
    idf, _ = semantic.compute_idf(iter(texts))

    results = []
    honeypot_count = 0
    for c in candidates:
        if check_honeypot(c):
            honeypot_count += 1
            continue
        s = scoring.score_candidate(c, idf)
        results.append((s, c))

    results.sort(key=lambda x: (-x[0].score, x[1]["candidate_id"]))
    top = results[:top_n]

    rows = []
    for rank, (scored, candidate) in enumerate(top, start=1):
        rows.append({
            "rank": rank,
            "candidate_id": candidate["candidate_id"],
            "score": scored.score,
            "reasoning": build_reasoning(candidate, scored),
            "candidate": candidate,
            "scored": scored,
        })

    return rows, honeypot_count, jd


def company_tier_badge(tier: str) -> str:
    colors = {
        "ai_native":   ("", "#f0fdf4", "#166534"),
        "global_major":("", "#eff6ff", "#1d4ed8"),
        "product":     ("", "#f5f3ff", "#6d28d9"),
        "consulting":  ("", "#fff7ed", "#c2410c"),
        "generic":     ("", "#f8fafc", "#64748b"),
        "ambiguous":   ("", "#f8fafc", "#64748b"),
    }
    emoji, bg, fg = colors.get(tier, ("", "#f8fafc", "#64748b"))
    return f"<span style='background:{bg};color:{fg};border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:600'>{emoji} {tier}</span>"

with st.sidebar:
    st.image("https://img.shields.io/badge/Redrob-Ranker-2563eb?style=flat-square&logo=python", width=160)
    st.markdown("##  Settings")

    data_source = st.radio(
        "Candidate data",
        ["Use built-in 50-candidate sample", "Upload candidates.jsonl / .jsonl.gz"],
        index=0,
    )

    uploaded = None
    if data_source == "Upload candidates.jsonl / .jsonl.gz":
        uploaded = st.file_uploader(
            "Upload candidates.jsonl or .jsonl.gz",
            type=["jsonl", "gz"],
            help="The full 100k file (~487 MB) works fine — ranking runs on CPU only."
        )

    top_n = st.slider("Top N candidates to show", 10, 100, 100, step=10)

    st.markdown("---")
    st.markdown("###  Job Description")
    jd_path = str(ROOT / "data" / "job_description.md")
    if Path(jd_path).exists():
        with open(jd_path) as f:
            jd_text_preview = f.read()[:600]
        with st.expander("Preview JD"):
            st.markdown(jd_text_preview + "\n\n*…(truncated)*")
    st.caption("Ranking against: **Senior AI Engineer — Founding Team**")

    st.markdown("---")

st.markdown("# Redrob Candidate Ranker")
st.markdown("*Intelligent Candidate Discovery & Ranking — Senior AI Engineer, Founding Team*")



if data_source == "Use built-in 50-candidate sample":
    sample_path = ROOT / "data" / "sample_candidates.json"
    if not sample_path.exists():
        st.error("sample_candidates.json not found in data/. Please upload a candidates file.")
        st.stop()
    with open(sample_path) as f:
        raw_candidates = json.load(f)
    candidates_jsonl = "\n".join(json.dumps(c) for c in raw_candidates)
    st.info(f"Using built-in sample of **{len(raw_candidates)} candidates**. Upload candidates.jsonl for the full 100k run.")
else:
    if uploaded is None:
        st.warning("Upload a candidates.jsonl or .jsonl.gz file to begin.")
        st.stop()
    with st.spinner("Reading candidates file…"):
        raw_candidates = load_candidates_from_upload(uploaded)
    candidates_jsonl = "\n".join(json.dumps(c) for c in raw_candidates)
    st.success(f"Loaded **{len(raw_candidates):,} candidates** from upload.")


with st.spinner(f"Ranking {len(raw_candidates):,} candidates… (two-pass IDF + scoring)"):
    rows, honeypot_count, jd = run_ranking(candidates_jsonl, jd_path, top_n)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-number'>{len(raw_candidates):,}</div>
      <div class='metric-label'>Candidates scored</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-number'>{honeypot_count}</div>
      <div class='metric-label'>Honeypots excluded</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-number'>{len(rows)}</div>
      <div class='metric-label'>Top candidates shown</div>
    </div>""", unsafe_allow_html=True)
with col4:
    top_score = rows[0]["score"] if rows else 0
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-number'>{top_score:.3f}</div>
      <div class='metric-label'>#1 score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

tab1, tab2 = st.tabs([" Ranked Results", " Score Breakdown"])

with tab1:
    for r in rows:
        cand = r["candidate"]
        scored = r["scored"]
        prof = cand["profile"]
        comp = scored.components
        title_ev = comp["title_fit"].evidence
        skill_ev = comp["skill_match"].evidence
        gate_ev = comp["signal_gate"].evidence

        with st.expander(
            f"#{r['rank']:>3}  {prof['current_title']}  ·  {prof['current_company']}  ·  "
            f"{prof['years_of_experience']}y  ·  score {r['score']:.4f}",
            expanded=(r["rank"] <= 3),
        ):
            left, right = st.columns([3, 2])
            with left:
                
                st.markdown(
                    f"<span class='rank-badge'>#{r['rank']}</span>  "
                    f"<span class='score-badge'>{r['score']:.4f}</span>  "
                    f"{company_tier_badge(title_ev.get('company_tier','generic'))}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {prof['current_title']}")
                st.markdown(f"**{prof['current_company']}** · {prof['years_of_experience']} yrs · {prof.get('location','?')}")

                must = skill_ev.get("matched_must", [])
                nice = skill_ev.get("matched_nice", [])
                adj  = skill_ev.get("matched_adjacent", [])
                watch = skill_ev.get("matched_watch", [])
                if must or nice or adj:
                    st.markdown("**Matched skills:**")
                    tags = (
                        "".join(f"<span class='tag tag-must'>✓ {s}</span>" for s in must) +
                        "".join(f"<span class='tag tag-nice'>+ {s}</span>" for s in nice) +
                        "".join(f"<span class='tag'>{s}</span>" for s in adj) +
                        "".join(f"<span class='tag tag-warn'>👁 {s}</span>" for s in watch)
                    )
                    st.markdown(tags, unsafe_allow_html=True)

        
                st.markdown(f"**Reasoning:** *{r['reasoning']}*")

               
                if scored.disqualifier_reasons:
                    dq = scored.disqualifier_reasons[0].split("]", 1)[-1].strip()
                    st.markdown(f"<div class='caveat-box'> {dq}</div>", unsafe_allow_html=True)

            with right:
             
                st.markdown("**Component scores:**")
                components_display = [
                    ("Title fit",      comp["title_fit"].score,      "0.22"),
                    ("Skill match",    comp["skill_match"].score,     "0.28"),
                    ("Semantic",       comp["semantic_match"] if isinstance(comp["semantic_match"], float) else comp["semantic_match"], "0.08"),
                    ("Experience",     comp["experience_fit"].score,  "0.12"),
                    ("Location",       comp["location_fit"].score,    "0.10"),
                    ("Education",      comp["education_fit"].score,   "0.05"),
                    ("Validation",     comp["validation_fit"].score,  "0.05"),
                    ("Stability",      comp["career_stability"].score,"0.10"),
                ]
                for label, val, weight in components_display:
                    val_f = val if isinstance(val, float) else float(val)
                    c_col = component_color(val_f)
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;margin-top:6px'>"
                        f"<span style='color:#374151'>{label} <span style='color:#9ca3af'>×{weight}</span></span>"
                        f"<span style='font-weight:600;color:{c_col};font-family:monospace'>{val_f:.2f}</span>"
                        f"</div>{score_bar(val_f, c_col)}",
                        unsafe_allow_html=True,
                    )

                gate_val = comp["signal_gate"].score
                dq_mult = comp["dq_multiplier"]
                st.markdown(
                    f"<div style='margin-top:10px;font-size:0.8rem;border-top:1px solid #e5e7eb;padding-top:8px'>"
                    f"<b>Signal gate:</b> <span style='font-family:monospace;color:{'#16a34a' if gate_val>0.8 else '#d97706'}'>{gate_val:.2f}×</span>"
                    f"&nbsp;&nbsp;<b>DQ mult:</b> <span style='font-family:monospace;color:{'#dc2626' if dq_mult<0.9 else '#16a34a'}'>{dq_mult:.2f}×</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                sig = cand.get("redrob_signals", {})
                st.markdown(
                    f"<div style='margin-top:8px;font-size:0.78rem;color:#6b7280'>"
                    f" Last active: {gate_ev.get('last_active_date','?')} &nbsp;|&nbsp; "
                    f" Response rate: {sig.get('recruiter_response_rate',0):.0%} &nbsp;|&nbsp; "
                    f" Notice: {sig.get('notice_period_days','?')}d"
                    f"</div>",
                    unsafe_allow_html=True,
                )

with tab2:
    try:
        import json as _json

        scores = [r["score"] for r in rows]
        titles = [r["candidate"]["profile"]["current_title"] for r in rows]
        companies = [r["candidate"]["profile"]["current_company"] for r in rows]

        st.markdown("### Score distribution — top candidates")

        import collections
        title_dist = collections.Counter(titles)
        st.markdown("**Title breakdown in this ranking:**")
        title_df_data = {t: c for t, c in title_dist.most_common(15)}
        st.bar_chart(title_df_data)

        from src.ontology import company_tier as ctier
        tier_dist = collections.Counter(ctier(c) for c in companies)
        st.markdown("**Company tier breakdown:**")
        st.bar_chart(dict(tier_dist))

        st.markdown("**Score distribution across ranked candidates:**")
        score_buckets = collections.Counter()
        for s in scores:
            bucket = f"{int(s*10)/10:.1f}"
            score_buckets[bucket] += 1
        st.bar_chart(dict(sorted(score_buckets.items())))

    except Exception as e:
        st.warning(f"Chart rendering issue: {e}")

st.divider()
st.markdown("###  Download submission CSV")

csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(["candidate_id", "rank", "score", "reasoning"])
for r in rows:
    writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.8f}", r["reasoning"]])

st.download_button(
    label=f"⬇️ Download top-{len(rows)} submission.csv",
    data=csv_buffer.getvalue(),
    file_name="submission.csv",
    mime="text/csv",
)

st.caption(
    "Built for the Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge. "
    "Rule-based ranker with ontology-aware skill matching, trust-adjusted scoring, "
    "and behavioral signal gating. Zero ML dependencies at inference time."
)
