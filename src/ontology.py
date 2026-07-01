"""
Domain knowledge layer: skill canonicalization, company tiers, title taxonomy.

These mappings were derived by analyzing the actual candidate pool
(100k records) rather than guessed from the JD text alone. In particular:

  - The skill vocabulary in this dataset has 133 unique skill strings.
    A cluster of ~46 "AI/ML core" skills appears at low-to-moderate
    frequency (roughly 1,300-5,200 occurrences); a cluster of ~87
    "generic" skills (sales, accounting, frontend frameworks, etc.)
    appears at high uniform frequency (~11,800-12,250). Within the
    AI-core cluster, a handful of skill *strings* appear only 1-7 times
    total (e.g. "Natural Language Processing" vs "NLP", "Ranking
    Systems" vs "Learning to Rank", "Text Encoders" vs "Embeddings").
    These low-frequency strings are paraphrases of the more common
    canonical terms — almost certainly inserted to test whether a
    ranker does synonym-aware matching or pure exact-keyword matching.
    SKILL_SYNONYMS below maps every observed surface form to one of a
    small number of canonical skill IDs.

  - The 63 unique employer names in the dataset cluster into clear
    tiers by frequency and real-world identity: fictional generic
    placeholders (Wayne Enterprises, Hooli, Pied Piper, ...), large
    Indian IT-services/consulting majors, recognizable Indian product
    companies, a distinct tier of real Indian AI-native startups
    (Sarvam AI, Krutrim, Yellow.ai, ...), and rare global tech majors.
    COMPANY_TIERS encodes this.

Update these tables if the underlying dataset/JD changes; nothing else
in the pipeline should need to change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Skill canonicalization
# ---------------------------------------------------------------------------
# Canonical skill IDs group surface forms that mean the same capability.
# Every skill string actually observed in candidates.jsonl is mapped here
# (run scripts/inspect_skill_vocab.py to regenerate/verify against a new
# dataset dump).

SKILL_SYNONYMS: dict[str, str] = {
    # --- NLP / language ---
    "NLP": "nlp",
    "Natural Language Processing": "nlp",
    "Machine Learning": "ml_general",
    "Deep Learning": "deep_learning",
    "Data Science": "data_science",
    "Statistical Modeling": "statistical_modeling",
    "Feature Engineering": "feature_engineering",
    "Forecasting": "forecasting",
    "Time Series": "time_series",
    "Reinforcement Learning": "reinforcement_learning",
    # --- Embeddings / retrieval / vector search (core to this JD) ---
    "Embeddings": "embeddings",
    "Text Encoders": "embeddings",
    "Vector Representations": "embeddings",
    "Sentence Transformers": "embeddings",
    "Vector Search": "vector_search",
    "Semantic Search": "vector_search",
    "Information Retrieval": "information_retrieval",
    "Information Retrieval Systems": "information_retrieval",
    "Search & Discovery": "information_retrieval",
    "Search Backend": "information_retrieval",
    "Search Infrastructure": "information_retrieval",
    "Indexing Algorithms": "information_retrieval",
    "BM25": "lexical_retrieval",
    "Learning to Rank": "ranking",
    "Ranking Systems": "ranking",
    "Recommendation Systems": "ranking",
    "Content Matching": "ranking",
    # --- Vector DBs / hybrid search infra ---
    "FAISS": "vector_db",
    "Pinecone": "vector_db",
    "Qdrant": "vector_db",
    "Milvus": "vector_db",
    "Weaviate": "vector_db",
    "OpenSearch": "vector_db",
    "Elasticsearch": "vector_db",
    "pgvector": "vector_db",
    # --- LLMs ---
    "LLMs": "llms",
    "Prompt Engineering": "llms",
    "RAG": "rag",
    "Document Processing": "rag",
    "Fine-tuning LLMs": "fine_tuning",
    "Model Adaptation": "fine_tuning",
    "LoRA": "fine_tuning",
    "QLoRA": "fine_tuning",
    "PEFT": "fine_tuning",
    "Hugging Face Transformers": "llm_tooling",
    "LangChain": "llm_tooling",
    "LlamaIndex": "llm_tooling",
    "Haystack": "llm_tooling",
    "Open-source ML libraries": "llm_tooling",
    # --- MLOps ---
    "MLOps": "mlops",
    "MLflow": "mlops",
    "Kubeflow": "mlops",
    "BentoML": "mlops",
    "Weights & Biases": "mlops",
    "Workflow Orchestration": "mlops",
    # --- Core ML frameworks / languages ---
    "Python": "python",
    "PyTorch": "ml_framework",
    "TensorFlow": "ml_framework",
    "scikit-learn": "ml_framework",
    # --- Computer vision / speech (adjacent, not core per JD) ---
    "Computer Vision": "cv",
    "CNN": "cv",
    "Image Classification": "cv",
    "Object Detection": "cv",
    "YOLO": "cv",
    "OpenCV": "cv",
    "GANs": "generative_cv",
    "Diffusion Models": "generative_cv",
    "Speech Recognition": "speech",
    "ASR": "speech",
    "TTS": "speech",
}

# Skills that exist in candidates.jsonl but are generic/non-AI; explicitly
# left OUT of SKILL_SYNONYMS canonicalization (no canonical AI id) so they
# fall through to "unmatched" in scoring. Kept here only for documentation.
GENERIC_SKILLS_NOTE = (
    "HTML, CSS, JavaScript, TypeScript, React, Vue.js, Angular, Redux, "
    "Next.js, Tailwind, Webpack, Figma, Illustrator, Photoshop, Java, Go, "
    "Rust, Spring Boot, Django, Flask, FastAPI, Node.js, .NET-style infra "
    "skills, REST APIs, GraphQL, gRPC, Microservices, Docker, Kubernetes, "
    "Terraform, CI/CD, AWS, GCP, Azure, Databricks, Snowflake, BigQuery, "
    "Airflow, Apache Beam, Apache Flink, Kafka, Spark, Hadoop, dbt, ETL, "
    "Data Pipelines, SQL, PostgreSQL, MongoDB, Redis, Sales, Marketing, "
    "SEO, Accounting, Tally, SAP, Salesforce CRM, Excel, PowerPoint, "
    "Project Management, Agile, Scrum, Six Sigma, Content Writing."
)

# Canonical skill -> human-readable label, and which JD bucket it belongs to.
# "must": explicitly required by the JD ("things you absolutely need").
# "nice": explicitly a plus but not required.
# "adjacent": related ML skill, not explicitly asked for, small credit.
# "watch": present but the JD flags it as a potential red flag if it's
#          *all* the candidate has (see disqualifiers.py).
CANONICAL_SKILLS: dict[str, dict] = {
    "embeddings":            {"label": "embeddings",                 "bucket": "must"},
    "vector_search":         {"label": "vector/semantic search",     "bucket": "must"},
    "vector_db":             {"label": "vector database",            "bucket": "must"},
    "information_retrieval": {"label": "information retrieval",      "bucket": "must"},
    "lexical_retrieval":     {"label": "lexical retrieval (BM25)",   "bucket": "must"},
    "ranking":               {"label": "ranking/recommendation systems", "bucket": "must"},
    "rag":                   {"label": "RAG",                        "bucket": "nice"},
    "llms":                  {"label": "LLMs",                       "bucket": "nice"},
    "llm_tooling":           {"label": "LLM tooling (LangChain etc.)", "bucket": "watch"},
    "fine_tuning":           {"label": "LLM fine-tuning",            "bucket": "nice"},
    "mlops":                 {"label": "MLOps",                      "bucket": "nice"},
    "python":                {"label": "Python",                     "bucket": "must"},
    "ml_framework":          {"label": "ML framework (PyTorch/TF/sklearn)", "bucket": "adjacent"},
    "nlp":                   {"label": "NLP",                        "bucket": "adjacent"},
    "ml_general":            {"label": "machine learning",           "bucket": "adjacent"},
    "deep_learning":         {"label": "deep learning",              "bucket": "adjacent"},
    "data_science":          {"label": "data science",               "bucket": "adjacent"},
    "statistical_modeling":  {"label": "statistical modeling",       "bucket": "adjacent"},
    "feature_engineering":   {"label": "feature engineering",        "bucket": "adjacent"},
    "forecasting":           {"label": "forecasting",                "bucket": "adjacent"},
    "time_series":           {"label": "time series modeling",       "bucket": "adjacent"},
    "reinforcement_learning":{"label": "reinforcement learning",     "bucket": "adjacent"},
    "cv":                    {"label": "computer vision",            "bucket": "watch"},
    "generative_cv":         {"label": "generative vision (GANs/diffusion)", "bucket": "watch"},
    "speech":                {"label": "speech/ASR/TTS",             "bucket": "watch"},
}

# "LLM tooling" (LangChain/LlamaIndex/Haystack/HF Transformers) is
# deliberately bucketed "watch" rather than "must"/"nice": the JD explicitly
# warns against candidates whose AI experience is recent (<12mo) LangChain-
# wrapper work with no underlying production ranking/retrieval experience.
# It still contributes some credit, but is down-weighted relative to the
# systems-level skills, and is the trigger for the "recent tool-calling
# only" disqualifier check in disqualifiers.py.

MUST_HAVE_IDS = {k for k, v in CANONICAL_SKILLS.items() if v["bucket"] == "must"}
NICE_TO_HAVE_IDS = {k for k, v in CANONICAL_SKILLS.items() if v["bucket"] == "nice"}
ADJACENT_IDS = {k for k, v in CANONICAL_SKILLS.items() if v["bucket"] == "adjacent"}
WATCH_IDS = {k for k, v in CANONICAL_SKILLS.items() if v["bucket"] == "watch"}

CV_SPEECH_IDS = {"cv", "generative_cv", "speech"}
NLP_IR_IDS = {"nlp", "embeddings", "vector_search", "information_retrieval",
              "lexical_retrieval", "ranking", "rag", "llms"}

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}


def canonical_skill(name: str) -> str | None:
    """Map a raw skill string to its canonical id, or None if unrecognized."""
    return SKILL_SYNONYMS.get(name.strip())


# ---------------------------------------------------------------------------
# Company tiers
# ---------------------------------------------------------------------------
# Derived from frequency + real-world identity of the 63 employer names
# actually present in candidates.jsonl.

CONSULTING_SERVICES_FIRMS = {
    # Explicitly named in the JD ("TCS, Infosys, Wipro, Accenture, Cognizant,
    # Capgemini, etc.") plus other large Indian IT-services majors that are
    # the same real-world category (the JD's "etc." implies this is not an
    # exhaustive list) and that appear in the dataset.
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis", "Mindtree",
}

AI_NATIVE_COMPANIES = {
    # Real Indian AI-focused startups present in the dataset. Strongest
    # "product company doing real AI work" signal available.
    "Sarvam AI", "Aganitha", "Rephrase.ai", "Niramai", "Glance", "Haptik",
    "Wysa", "Krutrim", "Saarthi.ai", "Verloop.io", "Mad Street Den",
    "Yellow.ai", "Locobuzz", "Observe.AI",
}

RECOGNIZED_PRODUCT_COMPANIES = {
    # Real, recognizable consumer/product tech companies (non-AI-specific
    # but unambiguously "product company" rather than services/consulting).
    "Swiggy", "CRED", "Razorpay", "Zomato", "Flipkart", "Meesho", "InMobi",
    "Nykaa", "Zoho", "Freshworks", "Vedantu", "Ola", "Paytm", "BYJU'S",
    "upGrad", "PolicyBazaar", "Dream11", "PharmEasy", "PhonePe",
    "Unacademy",
}

GLOBAL_TECH_MAJORS = {
    "Meta", "Google", "Netflix", "Amazon", "Microsoft", "Salesforce",
    "LinkedIn", "Apple", "Adobe", "Uber",
}

# "Genpact AI" sits at low frequency (42) alongside the AI-native tier, but
# the parent Genpact brand is, in the real world, a BPO/analytics services
# company. We treat it as ambiguous: not penalized as consulting-only (the
# "AI" branding and dataset frequency both diverge from the IT-services
# tier), but also not given the strong AI-native bonus. It nets to neutral.
AMBIGUOUS_COMPANIES = {"Genpact AI"}

# Fictional generic placeholders (Wayne Enterprises, Hooli, Pied Piper,
# Stark Industries, Initech, Globex Inc, Acme Corp, Dunder Mifflin) carry
# no real-world signal by name. They are deliberately NOT special-cased
# here; candidates at these employers are scored purely on company_size /
# industry / role content, which is the correct behavior for a synthetic
# generic-employer name.


def company_tier(name: str) -> str:
    """Classify a company name into one of: consulting, ai_native,
    product, global_major, ambiguous, generic."""
    if name in CONSULTING_SERVICES_FIRMS:
        return "consulting"
    if name in AI_NATIVE_COMPANIES:
        return "ai_native"
    if name in RECOGNIZED_PRODUCT_COMPANIES:
        return "product"
    if name in GLOBAL_TECH_MAJORS:
        return "global_major"
    if name in AMBIGUOUS_COMPANIES:
        return "ambiguous"
    return "generic"


# ---------------------------------------------------------------------------
# Title taxonomy
# ---------------------------------------------------------------------------
# The dataset's 30 distinct current_title values split cleanly into three
# groups by frequency and content. GENERIC_NONTECH_TITLES is the ~69k-strong
# keyword-stuffer trap population: candidates with these titles who also
# carry many AI skill tags are the textbook case the JD explicitly warns
# about ("a candidate who has all the AI keywords listed as skills but
# whose title is 'Marketing Manager' is not a fit").

GENERIC_NONTECH_TITLES = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager",
    "Content Writer", "Sales Executive", "Civil Engineer",
    "Graphic Designer", "Marketing Manager",
}

GENERAL_TECH_TITLES = {
    "Software Engineer", "Full Stack Developer", "Cloud Engineer",
    "Java Developer", ".NET Developer", "DevOps Engineer",
    "Mobile Developer", "Frontend Engineer", "QA Engineer",
}

DATA_ADJACENT_TITLES = {
    # Not explicitly "AI" titles, but exactly the kind of role the JD's
    # closing note describes: may show real ranking/retrieval/ML production
    # work in career_history despite a non-AI title.
    "Analytics Engineer", "Data Engineer", "Data Analyst",
    "Backend Engineer", "Senior Data Engineer", "Senior Software Engineer",
}

AI_CORE_TITLES = {
    "ML Engineer", "AI Research Engineer", "Data Scientist",
}

# Phrases that, if present in a career_history description, indicate real
# production ranking/retrieval/recommendation/ML system work regardless of
# the role's title. Used to surface Tier-5 "plain language" fits.
PRODUCTION_SYSTEM_PHRASES = [
    "recommendation system", "recommender system", "search ranking",
    "search relevance", "ranking model", "retrieval system",
    "retrieval-augmented", "semantic search", "vector search",
    "search infrastructure", "personalization", "feature pipeline",
    "real-time inference", "production model", "deployed model",
    "a/b test", "ab test", "click-through", "ctr", "relevance model",
    "query understanding", "embedding pipeline", "served to users",
    "millions of users", "at scale", "matching engine",
]

# Title-ladder tokens used for the "title-chaser" trajectory check.
TITLE_LADDER = ["junior", "associate", "engineer", "senior", "lead",
                "staff", "principal", "director", "vp", "head"]

ARCHITECT_NONCODING_TITLE_TOKENS = ["architect", "director", "vp", "head of", "tech lead", "engineering manager"]
RESEARCH_ONLY_TITLE_TOKENS = ["research scientist", "researcher", "postdoc", "research fellow"]
