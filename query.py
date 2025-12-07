import os
import numpy as np
import requests
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client.models import NamedVector
from qdrant_client.models import Query
import numpy as np

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_docs")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Connect to Qdrant ---
client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    )


# --- Embedding helper ---
def embed_text(text):
    payload = {"model": "nomic-embed-text", "input": text}
    r = requests.post("http://localhost:11434/api/embed", json=payload)
    r.raise_for_status()
    data = r.json()
    embedding = data.get("embeddings", [])
    return np.array(embedding, dtype=np.float32)


# --- Semantic Search ---
def search(query, top_k=5, rerank=False, min_relevance=0.5):
    """
    Semantic search with intelligent thresholding to ignore irrelevant results.
    """
    query_vector = embed_text(query)

    # Flatten the embedding if nested
    if hasattr(query_vector, "__len__") and len(query_vector) == 1:
        query_vector = query_vector[0]
    vector_data = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

    # Query Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_data,
        limit=top_k * 2 if rerank else top_k,
        with_payload=True
    )

    results = response.points
    if not results:
        print("⚠️ No results found in Qdrant.")
        return []

    # Inspect scores
    print("\n🔍 Raw scores:")
    for r in results:
        print(f"{r.payload.get('filename', 'N/A')} → {r.score:.4f}")

    # --- Intelligent filtering ---
    # Get highest score
    max_score = max(r.score for r in results)

    # If all are too low (below min_relevance), return nothing
    if max_score < min_relevance:
        print(f"🚫 No relevant results (max score={max_score:.3f} < {min_relevance})")
        return []

    # Use a dynamic threshold slightly below top score
    threshold = max(min_relevance, max_score - 0.15)
    filtered_results = [r for r in results if r.score >= threshold]

    # Optional reranking
    if rerank:
        filtered_results = rerank_results(query, filtered_results)

    # --- Final check ---
    if not filtered_results:
        print(f"🚫 No results above threshold={threshold:.3f}")
        return []

    print(f"✅ Returning {len(filtered_results)} results (threshold={threshold:.3f})")
    return filtered_results

def rerank_results(query_text, results, embed_fn=None):
    """
    Optionally rerank Qdrant search results using cosine similarity between 
    query and candidate texts.

    Args:
        query_text (str): Original query string.
        results (list): List of Qdrant ScoredPoint objects.
        embed_fn (callable): Function that embeds text into vectors. 
                             Should return a numpy array.

    Returns:
        list: Reranked results (sorted by semantic similarity).
    """
    if embed_fn is None:
        from query import embed_text  # import your existing embed function
        embed_fn = embed_text

    # Embed the query once
    query_vector = embed_fn(query_text)
    if isinstance(query_vector, list):
        query_vector = np.array(query_vector)
    query_vector = query_vector.flatten()

    # Compute new similarity scores
    rescored = []
    for r in results:
        doc_text = r.payload.get("text", "")
        doc_vector = embed_fn(doc_text)
        if isinstance(doc_vector, list):
            doc_vector = np.array(doc_vector)
        doc_vector = doc_vector.flatten()

        # cosine similarity
        sim = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        )
        rescored.append((r, sim))

    # Sort by similarity (descending)
    rescored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in rescored]

# ✅ Export alias for external imports (important!)
semantic_search = search


# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    hits = semantic_search(args.query, top_k=args.top_k)
    for r in hits:
        print(r.payload["filename"], ":", r.payload["text"][:150])
