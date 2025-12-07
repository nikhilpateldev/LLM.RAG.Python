import os
import textwrap
from typing import List

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import numpy as np

# --- CONFIG ---

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY","")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_docs")

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")


# --- CLIENTS ---
print(QDRANT_URL)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# --- HELPERS ---

def embed_text(text):
    print("1.2")
    payload = {"model": "nomic-embed-text", "input": text}
    r = requests.post("http://localhost:11434/api/embed", json=payload, timeout=60)
    print("1.3")
    r.raise_for_status()
    resp = r.json()
    print("debug")
    # pdb.set_trace()
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    return resp



def retrieve_context(question: str, top_k: int = 5) -> List[qmodels.ScoredPoint]:
    """
    Semantic search in Qdrant to retrieve top_k relevant chunks.
    """
    query_vector = embed_text(question)

    # ensure flat list, not [[...]]
    if hasattr(query_vector, "__len__") and len(query_vector) == 1:
        query_vector = query_vector[0]
    vector_data = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

    # Query Qdrant
    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_data,
        limit=5,
        with_payload=True
    )


    return res.points


def build_prompt(question: str, contexts: List[qmodels.ScoredPoint]) -> str:
    """
    Build a prompt for Llama3 using retrieved context chunks.
    """
    context_blocks = []
    print("Debug prompts")
    for i, pt in enumerate(contexts, start=1):
        payload = pt.payload or {}
        filename = payload.get("filename", "unknown")
        text = payload.get("text", "")
        context_blocks.append(
            f"[Document {i} | {filename} | score={pt.score:.3f}]\n{text}"
        )
    
    context_str = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say that you do not know and do NOT hallucinate.

Context:
{context_str}

Question:
{question}

Answer clearly and concisely:
""".strip()

    return prompt


def call_llm(prompt: str) -> str:
    """
    Call Llama3 via Ollama's /api/chat endpoint.
    """
    print("Debug")
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a factual assistant. Use only the provided context."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()

    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()

    return str(data)


def answer_question(question: str, top_k: int = 5, show_sources: bool = True) -> dict:
    """
    Full RAG pipeline:
      1. Embed question
      2. Retrieve context from Qdrant
      3. Build prompt
      4. Call Llama3
      5. Return answer + sources
    """
    points = retrieve_context(question, top_k=top_k)
    if not points:
        return {
            "answer": "I could not find any relevant information in the knowledge base.",
            "sources": []
        }

    prompt = build_prompt(question, points)
    answer = call_llm(prompt)

    if show_sources:
        sources = [
            {
                "id": p.id,
                "score": p.score,
                "filename": (p.payload or {}).get("filename"),
                "snippet": textwrap.shorten((p.payload or {}).get("text", ""), width=200, placeholder="...")
            }
            for p in points
        ]
    else:
        sources = []

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # Simple CLI test
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do I configure IIS on Windows?"
    result = answer_question(q, top_k=5, show_sources=True)

    print("\n🧠 Answer:\n")
    print(result["answer"])
    print("\n📚 Sources:\n")
    for src in result["sources"]:
        print(f"- {src['filename']} (score={src['score']:.3f})")
        print(f"  {src['snippet']}\n")
