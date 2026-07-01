# Redrob Hackathon — Candidate Ranker

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role being ranked for:** Senior AI Engineer — Founding Team (Redrob AI)

## Quickstart

```bash
# No dependencies to install — pure Python stdlib at rank time.
python rank.py \
  --candidates ./candidates.jsonl \
  --jd ./data/job_description.md \
  --out ./submission.csv

# Validate format before uploading
python validate_submission.py submission.csv
```

Runtime on 100k candidates: ~53s on a single CPU core, <4 GB RAM.

## How it works

### Core principle

The job description explicitly warns against two failure modes:
1. **Keyword-stuffer trap** — candidates with no real AI experience who list every
   AI buzzword as a skill (e.g. an HR Manager with "NLP, LLMs, RAG" self-tagged at
   "advanced" proficiency). The sample_submission.csv provided in the bundle is
   *exactly* this failure: it ranks an HR Manager #1 by raw keyword count.
2. **Plain-language Tier-5 miss** — a strong candidate (e.g. "Backend Engineer at Swiggy"
   who built a real semantic-search system) who never used the exact words "RAG" or
   "Pinecone" in their profile.

The ranker is explicitly designed to avoid both.

### Five scoring layers

**1. Title / career tier (weight 0.22)**  
The candidate's current title is the strongest single signal. The 47 unique titles in
the dataset split cleanly into: generic non-tech (Business Analyst, HR Manager etc. —
the keyword-stuffer population), general tech, data-adjacent (Backend Engineer, Data
Analyst — may be legitimate Tier-5 fits), and AI/ML-specific (ML Engineer,
Recommendation Systems Engineer, Search Engineer etc.). Non-tech titles start at a
0.05 base score. A production-signal bonus (+0.35) is awarded for career descriptions
containing real production-system language (ranking models, A/B tests, feature
pipelines, deployment at scale etc.) regardless of title.

**2. Skill match with ontology-aware synonym resolution (weight 0.28)**  
133 unique skill strings in the dataset resolve to ~25 canonical skill IDs. The
`embeddings` canonical ID, for example, covers "Embeddings", "Sentence Transformers",
"Text Encoders", and "Vector Representations" — four distinct surface strings that
all appear at very different frequencies (1–5,200 occurrences), clearly designed to
test synonym-aware matching. Each skill's contribution is multiplied by a
**trust factor**: if the candidate's claimed proficiency is substantially higher than
their platform assessment score for that skill, the contribution is down-weighted
proportionally. Must-have skills (embeddings, vector search/DB, information retrieval,
BM25, learning-to-rank, Python) carry 62% of the total skill score; nice-to-have
(RAG, LLMs, fine-tuning, MLOps) carry 25%.

**3. Semantic text match (weight 0.08)**  
Corpus-wide IDF-weighted phrase scoring over a curated vocabulary of ~45
production-system phrases ("offline-to-online correlation", "embedding drift",
"hybrid retrieval", etc.) computed in one streaming pass over the full pool — no
model, no network. Catches Tier-5 candidates whose career descriptions contain
authentic production-system language without ever using the exact JD skill keywords.

**4. Experience, location, education, validation, career stability (weights 0.12/0.10/0.05/0.05/0.10)**  
Piecewise-linear experience scoring centered on the JD's stated ideal (6-8 years).
Location scores 1.0 for Pune/Noida, 0.85 for explicitly welcomed cities (Hyderabad,
Mumbai, Delhi NCR), 0.55 for elsewhere-in-India with relocation willingness. GitHub
activity score + certification count form an external-validation bonus. Average
tenure 18-48 months scores 1.0 for the stability component.

**5. Behavioral signal gate (multiplicative)**  
A multiplicative gate [0.30, 1.12] on the base fit score, combining last-active
recency, open-to-work flag, recruiter response rate, interview completion rate,
offer acceptance rate, profile verification, notice period fit, and completeness.
A perfect-fit candidate who has been inactive for 6+ months scores 0.30× of their
base score — still visible in the ranking, but suppressed below active, equivalent
candidates.

### Disqualifiers

Seven anti-patterns from the JD map to multiplicative penalties (not hard
exclusions, since the JD hedges most with "probably"):

| Pattern | Multiplier |
|---|---|
| Entire career at IT-services/consulting only (no product company) | 0.18× |
| Pure research background, no production deployment language | 0.12× |
| Architecture/leadership title, not hands-on coding for 18+ months | 0.40× |
| Title-chaser pattern (≥3 quick-promotion stints ≤18 months) | 0.45× |
| CV/speech skills only, no NLP/IR overlap | 0.30× |
| Recent LangChain-only AI (<12 months) with no pre-LLM ML depth | 0.35× |
| 5+ years experience, zero GitHub, no certifications | 0.85× |

### Honeypot exclusion

~70 candidates in the pool contain internal impossibilities (duration_months
inconsistent with date arithmetic, career time sums wildly mismatching stated
years_of_experience, or skills claimed at expert level with 0 months of use).
These are hard-excluded before scoring (not merely down-scored) to stay well
under the >10% honeypot-in-top-100 disqualification threshold.

## Repository structure

```
rank.py                  # single entry point
validate_submission.py   # official format validator (unmodified from bundle)
requirements.txt         # stdlib only at rank time
src/
  config.py              # all weights / thresholds
  ontology.py            # skill synonyms, company tiers, title taxonomy
  jd_parser.py           # structured JD requirement extractor + live cross-check
  honeypot.py            # internal-impossibility detector (calibrated to ~70/100k)
  disqualifiers.py       # 7 anti-pattern detectors
  semantic.py            # IDF phrase-match scorer
  features.py            # per-feature scoring functions + signal gate
  scoring.py             # composite score assembler
  reasoning.py           # grounded reasoning text generator
  io_utils.py            # streaming JSONL reader + CSV writer
data/
  job_description.md     # the role being ranked against
  candidate_schema.json  # field reference
  sample_candidates.json # first 50 candidates
  sample_submission.csv  # format reference (not a quality reference)
  # candidates.jsonl -- full pool, not committed (487 MB), pass via --candidates
```
