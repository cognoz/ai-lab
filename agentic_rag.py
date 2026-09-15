"""
Phase 1c — Agentic RAG with a fallback chain.

Builds on the merge of agent_loop.py (a model that calls tools in a loop)
and rag.py (retrieval over a Postgres knowledge base). Now the model has a
POLICY across multiple sources:

    1. Prefer the local knowledge base (authoritative, curated).
    2. If the KB doesn't contain the answer, fall back to web search
       (broad, but less trusted).
    3. Always state which source the answer came from.

Crucially, we do NOT hard-code "if no chunks then search web". That would
be the fixed-pipeline mindset. Instead we give the model BOTH tools and
express the policy in the system prompt — the model decides whether the
KB actually answered the question (which is subtler than a zero-length
check: the KB can return chunks that are present but irrelevant). This is
the agentic approach: the model orchestrates the fallback.

The web_search tool here is a STUB returning canned results, so you can
see the fallback routing without signing up for a search API. Swapping in
a real provider (Tavily/Brave/Bing/SerpAPI) means replacing the body of
web_search with an HTTP call — the agent logic doesn't change at all.

Run: OPENAI_API_KEY=sk-... DATABASE_URL=postgres://... python agentic_rag.py
"""
import json
import os
import psycopg
from openai import OpenAI

from rag import init_db, ingest, retrieve, SAMPLE
from agent_loop import get_weather, add

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# STUB web search. Returns canned results for a couple of known queries so
# the fallback path is observable. Anything else returns no results, which
# lets you see the model report "couldn't find it anywhere" honestly.
#
# TO GO REAL: replace the body with a call to a search API, e.g. Tavily:
#     resp = requests.post("https://api.tavily.com/search",
#                          json={"api_key": KEY, "query": query, "max_results": 3})
#     return {"source": "web", "query": query, "results": resp.json()["results"]}
# The tool's SHAPE (takes query, returns dict) stays identical, so nothing
# else in this file changes.
# ---------------------------------------------------------------------------
def web_search(query: str) -> dict:
    canned = {
        "istio ambient mesh": [
            "Istio ambient mode removes per-pod sidecars, using per-node "
            "ztunnel proxies for L4 and optional waypoint proxies for L7.",
        ],
        "kubernetes gateway api": [
            "The Kubernetes Gateway API is the successor to Ingress, with "
            "richer routing via Gateway and HTTPRoute resources.",
        ],
    }
    # naive keyword match, good enough to demo routing
    for key, results in canned.items():
        if any(word in query.lower() for word in key.split()):
            return {"source": "web", "query": query, "results": results,
                    "num_results": len(results)}
    return {"source": "web", "query": query, "results": [], "num_results": 0}


# ---------------------------------------------------------------------------
# Tools. search_knowledge_base closes over the DB connection (the model
# never sees it). Note both search tools now tag their results with a
# "source" field — that's what lets the model attribute the answer and lets
# YOU see, in the trace, which source won.
# ---------------------------------------------------------------------------
def build_tools(conn):
    def search_knowledge_base(query: str) -> dict:
        chunks = retrieve(conn, query, k=4)
        return {"source": "knowledge_base", "query": query,
                "chunks": chunks, "num_results": len(chunks)}

    return {
        "get_weather": get_weather,
        "add": add,
        "search_knowledge_base": search_knowledge_base,
        "web_search": web_search,
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the internal, authoritative DevOps knowledge base. "
                "ALWAYS try this FIRST for operational questions (Kubernetes, "
                "Kyverno, Cosign, Falco, spot pools, etc.). Returns the most "
                "relevant internal note chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Natural-language search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web. Use this ONLY as a fallback when "
                "search_knowledge_base did not return an answer, or for topics "
                "clearly outside the internal knowledge base. Web results are "
                "less authoritative than the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Natural-language search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a DevOps assistant with access to an internal knowledge base and "
    "a web-search fallback. Follow this policy:\n"
    "1. For operational/technical questions, ALWAYS call search_knowledge_base "
    "FIRST.\n"
    "2. Judge whether the returned chunks actually answer the question. If they "
    "do, answer from them and state: (source: knowledge base).\n"
    "3. If the knowledge base has no relevant answer, call web_search as a "
    "fallback. If it helps, answer from it and state: (source: web).\n"
    "4. If neither source answers, say so plainly. Do not invent an answer.\n"
    "5. For general questions (math, weather), use the appropriate tool or "
    "answer directly; you need not search."
)


def run(user_prompt: str, tools: dict, max_turns: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for _turn in range(max_turns):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for call in msg.tool_calls:
            fn = tools[call.function.name]
            args = json.loads(call.function.arguments)
            result = fn(**args)
            # compact trace so you SEE the routing + which source answered
            name = call.function.name
            if name in ("search_knowledge_base", "web_search"):
                preview = f"{result['num_results']} results for {result['query']!r}"
            else:
                preview = result
            print(f"  [tool] {name}({args}) -> {preview}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(result)}
            )

    return "(stopped: hit max_turns)"


if __name__ == "__main__":
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=False) as conn:
        init_db(conn)
        conn.execute("TRUNCATE chunks")
        conn.commit()
        ingest(conn, "lab-notes", SAMPLE)

        tools = build_tools(conn)

        prompts = [
            # 1. KB can answer -> KB only, "(source: knowledge base)"
            "How do I make Kyverno work with Cosign v3?",
            # 2. KB CANNOT answer, web stub CAN -> fallback to web
            "What is Istio ambient mesh?",
            # 3. Neither can answer -> honest "couldn't find it"
            "What is our internal policy on VPN access?",
            # 4. No search needed at all
            "What is 47 plus 55?",
        ]
        for p in prompts:
            print(f"\nUSER: {p}")
            print(f"ASSISTANT: {run(p, tools)}")
