"""
Rerank candidate hits:
 - Combines the vector score (as returned by Qdrant) and a lexical similarity score
 - Lexical score uses difflib.SequenceMatcher on lowercased text
 - Weighted sum: final_score = alpha * vector_score_norm + (1-alpha) * lexical_score
"""
from difflib import SequenceMatcher
import math

def lexical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def normalize_scores(hits):
    # normalize Qdrant scores to [0,1]
    scores = [h.score for h in hits]
    if not scores:
        return {}
    min_s, max_s = min(scores), max(scores)
    if math.isclose(min_s, max_s):
        return {h.id: 1.0 for h in hits}
    out = {}
    for h in hits:
        out[h.id] = (h.score - min_s) / (max_s - min_s)
    return out

def rerank_hits(query: str, query_vector, hits, top_k: int = 5, alpha: float = 0.7):
    """
    alpha -> weight for vector score (0..1)
    """
    vec_norm = normalize_scores(hits)
    scored = []
    for h in hits:
        lex = lexical_similarity(query, h.payload.get("text", ""))
        vec = vec_norm.get(h.id, 0.0)
        final = alpha * vec + (1 - alpha) * lex
        # attach final score for sorting and return
        h.final_score = final
        scored.append(h)
    scored.sort(key=lambda x: x.final_score, reverse=True)
    # return top_k items
    return scored[:top_k]
