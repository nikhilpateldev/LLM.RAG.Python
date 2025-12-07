"""
Ingest pipeline:
 - Load documents from data dir using data_loader.py
 - Chunk text into overlapping chunks
 - Batch-embed chunks via Ollama embedding endpoint
 - Upsert to Qdrant with chunk-level payloads (doc_id, filename, chunk_index, text)
"""
import os, math, requests, argparse, time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from data_loader import load_directory
import uuid
import pdb;
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embed")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """
    Simple chunker measured in characters (not tokens) for portability.
    Returns list of (chunk_text, start_idx)
    """
    if len(text) <= chunk_size:
        return [(text, 0)]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, start))
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def embed_texts(texts):
    payload = {"model": OLLAMA_MODEL, "input": texts}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    print("debug")
    # pdb.set_trace()
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    return resp

def ensure_collection(client: QdrantClient, dim: int):
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def main(data_dir: str, batch_size: int = 16, chunk_size: int = 800, overlap: int = 200):
    client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    )
    abs_path = os.path.join(os.path.dirname(__file__), data_dir)
    docs = load_directory(abs_path)
    print(f"Loaded {len(docs)} documents from {data_dir}")

    # prepare chunks
    points = []
    items = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, (chunk, start) in enumerate(chunks):
            cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['id']}_{i}"))
            items.append({"id": cid, "doc_id": doc["id"], "filename": doc["meta"].get("filename"), "text": chunk, "chunk_index": i})

    print(f"Total chunks: {len(items)}")
    if len(items) == 0:
        return

    # Batch embed & upsert
    # embed in batches to avoid giant requests
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        texts = [it["text"] for it in batch]
        embeddings = embed_texts(texts)
        dim = len(embeddings[0])
        # ensure collection once (on first batch)
        if i == 0:
            ensure_collection(client, dim)
        points = []
        for it, emb in zip(batch, embeddings):
            payload = {
                "doc_id": it["doc_id"],
                "filename": it["filename"],
                "text": it["text"],
                "chunk_index": it["chunk_index"],
            }
            points.append(PointStruct(id=it["id"], vector=emb, payload=payload))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted batch {i // batch_size + 1} ({len(points)} points)")
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="test_data", help="Directory containing pdf/html/txt files")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    main(args.data_dir, args.batch_size, args.chunk_size, args.overlap)
