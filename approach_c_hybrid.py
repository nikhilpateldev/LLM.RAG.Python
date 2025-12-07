
from rag import retrieve_context, build_prompt, call_llm, qdrant, COLLECTION_NAME
"""Hybrid retrieval improves recall by combining:

Vector search

Keyword/BM25 search

Metadata filtering (filename, category, tags)"""
def keyword_search(q,limit=5):
    """
    Basic keyword search scanning Qdrant payloads.
    (Not high-performance but works for demo.)
    """
    print(qdrant)
    pts,_ = qdrant.scroll(collection_name=COLLECTION_NAME,limit=500,with_payload=True)
    print("1")
    words=set(q.lower().split())
    print("2")
    scored=[]
    for p in pts:
        t=(p.payload or {}).get("text","").lower()
        score=sum(1 for w in words if w in t)
        if score>0: scored.append((score,p))
    scored.sort(reverse=True,key=lambda x:x[0])
    print("3")
    return [p for _,p in scored[:limit]]

def hybrid_retrieve(q,limit=5):
    print("1")
    v=retrieve_context(q,limit)
    print("2")
    k=keyword_search(q,limit)
    merged={p.id:p for p in v}
    for p in k: merged[p.id]=p
    return list(merged.values())[:limit]

def answer_hybrid(q):
    print("1.1")
    pts=hybrid_retrieve(q)
    prompt=build_prompt(q,pts)
    return call_llm(prompt), pts
