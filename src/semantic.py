"""
Lightweight semantic text-match layer.

This is NOT an embeddings model — it's a deliberately small, fully
auditable supplementary signal that catches candidates whose *free text*
(summary + career_history descriptions) describes real ranking/retrieval/
ML-production work without necessarily tagging it as a structured `skills`
entry. That's exactly the "Tier 5 plain-language" case the JD calls out:
"a Tier 5 candidate may not use the words 'RAG' or 'Pinecone'... but if
their career history shows they built a recommendation system... they're
a fit."

Design: corpus-wide IDF over a fixed, curated vocabulary of JD-distinctive
phrases (production-system language + retrieval/evaluation jargon), so a
candidate who happens to mention a rare, highly-specific phrase ("embedding
drift", "offline-to-online correlation") scores higher than one who only
hits common, lower-signal phrases. This is the standard TF-IDF principle
applied at the phrase level instead of the single-token level, computed
with nothing but stdlib `math` and `collections` — no model weights, no
network call, fully reproducible.

To swap in real embeddings later (e.g. sentence-transformers), replace
`score_candidate_text()` with a cosine-similarity lookup against a
precomputed candidate-embedding index and keep everything else the same;
the rest of the pipeline only depends on get a 0..1 float per candidate.
"""

from __future__ import annotations

import math

# Curated, JD-distinctive phrases. Deliberately phrase-level (not single
# words) so common words like "search" or "model" don't dominate.
SEMANTIC_VOCAB = [
    "recommendation system", "recommender system", "search ranking",
    "search relevance", "ranking model", "retrieval system",
    "retrieval-augmented", "semantic search", "vector search",
    "search infrastructure", "personalization", "feature pipeline",
    "real-time inference", "production model", "deployed model",
    "a/b test", "ab test", "click-through", "relevance model",
    "query understanding", "query expansion", "embedding pipeline",
    "served to users", "millions of users", "at scale", "matching engine",
    "embedding drift", "index refresh", "retrieval-quality regression",
    "offline-to-online correlation", "evaluation framework",
    "hybrid retrieval", "dense retrieval", "sparse retrieval",
    "cold start", "nearest-neighbor", "feature store", "model registry",
    "shadow deployment", "canary deployment", "vocabulary mismatch",
    "collaborative filtering", "content-based ranking", "learning-to-rank",
    "offline benchmark", "relevance labeling", "fraud-detection",
    "schema drift",
]


def candidate_text(candidate: dict) -> str:
    parts = [candidate.get("profile", {}).get("summary", "")]
    parts += [h.get("description", "") for h in candidate.get("career_history", [])]
    return " ".join(parts).lower()


def compute_idf(texts_iter, vocab: list[str] = SEMANTIC_VOCAB) -> dict[str, float]:
    """One streaming pass to compute document frequency -> idf for each
    vocab phrase. texts_iter: iterable of lowercase strings."""
    n = 0
    df = {phrase: 0 for phrase in vocab}
    for text in texts_iter:
        n += 1
        for phrase in vocab:
            if phrase in text:
                df[phrase] += 1
    return {phrase: math.log((n + 1) / (df[phrase] + 1)) + 1.0 for phrase in vocab}, n


def score_candidate_text(text: str, idf: dict[str, float]) -> float:
    """0..1 normalized score: sum of idf weights for phrases present,
    divided by the sum of the top-K idf weights (so a candidate hitting
    several rare phrases can reach ~1.0 without needing to hit all 45)."""
    hit_weight = sum(w for phrase, w in idf.items() if phrase in text)
    # Normalize against the sum of the highest few weights rather than all
    # of them -- hitting ~5 distinctive phrases should already mean "very
    # strong textual match", not require nearly all 45.
    top_k = sorted(idf.values(), reverse=True)[:6]
    denom = sum(top_k) or 1.0
    return max(0.0, min(1.0, hit_weight / denom))
