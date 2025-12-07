
from rag import retrieve_context, build_prompt, call_llm
"""
LLM Planner decides if retrieval is needed

Flow:

User enters query

LLM determines: "Is RAG needed?"

If yes → run your RAG pipeline

If no → LLM answers directly
"""
def needs_rag(question:str)->bool:
    prompt=f"""Decide if RAG needed. If yes reply RAG_REQUIRED else NO_RAG.
    Query:{question}"""
    resp=call_llm(prompt).strip().upper()
    return "RAG_REQUIRED" in resp

def answer_question_conditional(q):
    if needs_rag(q):
        pts=retrieve_context(q)
        prompt=build_prompt(q,pts)
        ans=call_llm(prompt)
        return ans, pts
    else:
        return call_llm(q), []
