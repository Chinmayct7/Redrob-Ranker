"""
All tunable weights and thresholds live here so the scoring logic in
scoring.py stays readable, and so the methodology summary can point at one
place. Weights are not fit to any ground truth (we don't have one) — they
encode the explicit and implicit priorities stated in job_description.md.
"""

# --- Composite weights (sum to 1.0) -----------------------------------
# Bumped vs. a naive "skills 60%, everything else 40%" split because the
# JD repeatedly states it cares more about title/trajectory truthfulness
# and behavioral availability than raw skill-tag coverage.
W_TITLE = 0.22
W_SKILL = 0.28
W_SEMANTIC = 0.08
W_EXPERIENCE = 0.12
W_LOCATION = 0.10
W_EDUCATION = 0.05
W_VALIDATION = 0.05
W_CAREER_STABILITY = 0.10

assert abs((W_TITLE + W_SKILL + W_SEMANTIC + W_EXPERIENCE + W_LOCATION
            + W_EDUCATION + W_VALIDATION + W_CAREER_STABILITY) - 1.0) < 1e-9

# --- Experience band (JD: "5-9 years... ideal is roughly 6-8") --------
EXP_IDEAL_MIN = 6.0
EXP_IDEAL_MAX = 8.0
EXP_ACCEPTABLE_MIN = 5.0
EXP_ACCEPTABLE_MAX = 9.0
EXP_HARD_FLOOR = 2.0   # below this, score decays steeply regardless of band

# --- Location (JD section "On location, comp, and logistics") --------
PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi ncr", "delhi", "gurgaon", "gurugram", "new delhi"}

# --- Notice period (JD: "We'd love sub-30-day notice... buy out up to 30") -
NOTICE_IDEAL_MAX_DAYS = 30
NOTICE_HARD_MAX_DAYS = 90

# --- Disqualifier / anti-pattern multipliers (applied multiplicatively) -
# "we will not move forward" (stated as an absolute) -> low multiplier.
# "we will probably not move forward" (stated as a strong default, not
# absolute) -> moderate multiplier, since the JD itself hedges with
# "unless you can demonstrate..." / "probably".
MULT_PURE_RESEARCH_NO_PRODUCTION = 0.12     # "we will not move forward"
MULT_CONSULTING_ONLY_CAREER = 0.18          # "we've had bad fit experiences in both directions"
MULT_RECENT_TOOLCALLING_ONLY = 0.35         # "probably not move forward, unless..."
MULT_ARCHITECT_NOT_CODING = 0.40            # "probably not move forward... this role writes code"
MULT_TITLE_CHASER = 0.45                    # "we're not a fit"
MULT_CV_SPEECH_ROBOTICS_ONLY = 0.30         # "you'd be re-learning fundamentals"
MULT_CLOSED_SOURCE_NO_VALIDATION = 0.85     # mild, JD doesn't hard-disqualify this

# Title-chaser thresholds.
TITLE_CHASER_MAX_TENURE_MONTHS = 18
TITLE_CHASER_MIN_JUMPS = 3

# --- Signal-quality gate bounds ---------------------------------------
# Multiplicative gate on top of the base fit score. Bounded so that even a
# fully "dark" but otherwise perfect-fit profile is suppressed, not erased
# (recruiters may still want to see them lower down, and a hard zero would
# make the score column look uniformly degenerate for non-active users).
SIGNAL_GATE_MIN = 0.30
SIGNAL_GATE_MAX = 1.12

LAST_ACTIVE_FULL_CREDIT_DAYS = 30
LAST_ACTIVE_ZERO_CREDIT_DAYS = 180

# --- Honeypot / impossibility checks -----------------------------------
DATE_DURATION_TOLERANCE_MONTHS = 3
EXPERTISE_MIN_MONTHS_FOR_EXPERT = 6
EXPERTISE_MIN_MONTHS_FOR_ADVANCED = 3
OVERLAP_TOLERANCE_MONTHS = 1
CAREER_SUM_VS_YOE_UPPER_RATIO = 1.35
CAREER_SUM_VS_YOE_LOWER_RATIO = 0.45

# --- Output ---------------------------------------------------------
TOP_N = 100
RANDOM_SEED = 1337  # only used for deterministic tie-break jitter avoidance, not for any stochastic modeling
