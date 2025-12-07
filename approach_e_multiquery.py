
from rag import retrieve_context, build_prompt, call_llm
"""This improves recall dramatically — LLM expands user query into multiple semantic queries.
The LLM rewrites the user question into multiple alternative search queries →
Each query is sent to the vector database →
Results are merged →
Reranked →
Sent to the LLM with the best context.

Example query from user:

“How do I renew my insurance policy after changing my address?”

A regular RAG pipeline embeds this ENTIRE question → searches →
Fails if your documents use different wording like:

“policy renewal procedure”

“address update during renewal”

“documentation needed for renewal”

Because embeddings still struggle with:

Multi-intent questions

Long compound questions

Niche industry vocabulary

Domain synonyms

Instead of embedding the single original question, we ask the LLM to generate 3–4 alternate semantic queries.

Example LLM Expansion:

1. "insurance policy renewal steps"
2. "address change impact on policy renewal"
3. "documents needed for address update during renewal"
"""
def expand_query(q):
    prompt=f"""Rewrite into 3 semantic queries. Output Python list.
    Q:{q}"""
    r=call_llm(prompt)
    try: return eval(r)
    except: return [q]

def multi_query_retrieve(q,top_k=5):
    qs=expand_query(q)
    print("Expanded queries:", qs)
    merged={}
    for sub in qs:
        pts=retrieve_context(sub,top_k)
        for p in pts: merged[p.id]=p
    return list(merged.values())[:top_k]

def answer_multi(q):
    pts=multi_query_retrieve(q)
    prompt=build_prompt(q,pts)
    return call_llm(prompt), pts
