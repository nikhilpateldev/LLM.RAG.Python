
from rag import retrieve_context, build_prompt, call_llm
"""This is an agent router, where LLM decides which tool to use."""
def route_query(q):
    prompt=f"""
You are a router that decides which tool is appropriate for the user's question.

Tools:
- RAG: For knowledge base lookups, documentation, policies, guides.
- SQL: For database-style queries (balance, status, counts, employee info).
- API: For external system requests (weather, live data, pricing).
- DIRECT: For simple conversational or reasoning questions.

Respond with ONLY one word: RAG, SQL, API, or DIRECT.

Question: {q}
"""
    r=call_llm(prompt).strip().upper()
    return r if r in ["RAG", "SQL", "API", "DIRECT"] else "RAG"

def answer_router(question):
    tool=route_query(question)
    print(f"Router selected: {tool}")

    if tool == "DIRECT":
        return {"answer": call_llm(question), "sources": []}

    if tool == "RAG":
        return build_prompt(question)

    if tool == "SQL":
        return {"answer": sql_tool(question), "sources": []}

    if tool == "API":
        return {"answer": api_tool(question), "sources": []}
    pts=retrieve_context(question)
    prompt=build_prompt(question,pts)
    return call_llm(prompt), pts

def sql_tool(question: str):
    return "SQL response example: (Simulated DB result)"

def api_tool(question: str):
    return "API response example: (Simulated external API)"