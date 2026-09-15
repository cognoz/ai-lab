"""
Phase 2a — Same agentic RAG, rebuilt on OpenAI's Responses API instead of
Chat Completions, using the REAL built-in web_search tool instead of the
Phase 1 stub.

WHY THIS FILE EXISTS (the comparison, not just a port):

  agentic_rag.py (Phase 1):
    - Chat Completions API
    - web_search is OUR code (a stub) — we own it, we control it, we pay
      nothing extra, but it's fake / limited to what we wrote.
    - Every tool is a "function tool": WE execute it and feed results back.

  responses_agentic_rag.py (this file):
    - Responses API
    - web_search is OpenAI's HOSTED tool — real live web access, OpenAI
      runs the search server-side, we never see the round trip. One line
      of config. But we get NO control over the index, sources, or
      freshness policy — it's a black box.
    - search_knowledge_base is STILL our own function tool, because it's
      OUR private Postgres data. OpenAI can't host that for us. This is
      the architectural point: generic capabilities (web search) CAN be
      built-in; your own data access NEVER can be.

STRUCTURAL DIFFERENCE FROM CHAT COMPLETIONS (this is the part that isn't
just a find-and-replace):

  Chat Completions                     Responses API
  ------------------------------------ --------------------------------
  messages list, role: assistant/tool  input list of ITEMS
  msg.tool_calls (on the message)      "function_call" items in output
  {"role":"tool", "tool_call_id":..}   {"type":"function_call_output",
                                         "call_id":...}
  YOU execute every tool               OpenAI executes built-in tools
                                        server-side; you only execute
                                        YOUR function tools

Run: OPENAI_API_KEY=sk-... DATABASE_URL=postgres://... python responses_agentic_rag.py
"""
import json
import os
import psycopg
from openai import OpenAI

from rag import init_db, ingest, retrieve, SAMPLE, connect_with_retry
from agent_loop import get_weather, add

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tools list. Note the MIX: one built-in ({"type": "web_search"}, no
# parameters, no description needed — OpenAI defines its own behaviour),
# and our own function tools using the Responses API's FLATTER schema
# (no nested "function" key, unlike Chat Completions).
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "web_search"},  # built-in, hosted, server-side. That's it.
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Search the internal, authoritative DevOps knowledge base. "
            "ALWAYS try this FIRST for operational questions (Kubernetes, "
            "Kyverno, Cosign, Falco, spot pools, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SYSTEM_PROMPT = (
    "You are a DevOps assistant. For operational/technical questions, call "
    "search_knowledge_base ONCE. Do not call it again with reworded or "
    "rephrased queries — one call is enough to know whether the knowledge "
    "base has the answer. If those results do not answer the question, "
    "immediately call web_search as a fallback — do not retry the knowledge "
    "base first. State which source you used: (source: knowledge base) or "
    "(source: web). If neither source answers the question, say so plainly "
    "rather than continuing to search. For math or weather, use the "
    "appropriate tool directly, no search needed."
)


def build_local_tools(conn):
    # ONLY our own function tools go here. web_search is never in this dict
    # — OpenAI executes it, we'd never see a call for it to dispatch.
    def search_knowledge_base(query: str) -> dict:
        chunks = retrieve(conn, query, k=4)
        return {"source": "knowledge_base", "query": query,
                "chunks": chunks, "num_results": len(chunks)}

    return {
        "search_knowledge_base": search_knowledge_base,
        "get_weather": get_weather,
        "add": add,
    }


def run(user_prompt: str, local_tools: dict, max_turns: int = 10, max_kb_calls: int = 2) -> str:
    # input is a flat list of items, not a messages-with-roles list.
    input_items = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    kb_calls_made = 0

    for _turn in range(max_turns):
        # THE ACTUAL FIX: once the cap is hit, stop OFFERING the tool at
        # all. Returning an error message and hoping the model reads it
        # was not a guarantee — we watched it get ignored eight times in
        # a row. If the tool isn't in the list, the model cannot call it,
        # full stop. This is the difference between telling the model
        # "please don't" and making the thing physically impossible.
        available_tools = TOOLS
        if kb_calls_made >= max_kb_calls:
            available_tools = [t for t in TOOLS if t.get("name") != "search_knowledge_base"]

        resp = client.responses.create(model=MODEL, input=input_items, tools=available_tools)
        input_items += resp.output

        function_calls = [item for item in resp.output if item.type == "function_call"]

        if not function_calls:
            return resp.output_text

        for call in function_calls:
            if call.name == "search_knowledge_base":
                kb_calls_made += 1
                # No need to block-and-message here anymore: once
                # kb_calls_made >= max_kb_calls, the NEXT loop iteration
                # simply won't offer this tool (see available_tools above),
                # so the model can't call it again regardless of what it
                # wants to do.

            fn = local_tools[call.name]
            args = json.loads(call.arguments)
            result = fn(**args)
            preview = result
            if call.name == "search_knowledge_base":
                preview = f"{result['num_results']} results for {result['query']!r}"
            print(f"  [local tool] {call.name}({args}) -> {preview}")
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })
        # loop again — model sees our results (and any built-in results
        # it already resolved) and continues

    return "(stopped: hit max_turns)"


if __name__ == "__main__":
    with connect_with_retry(os.environ["DATABASE_URL"], autocommit=False) as conn:
        init_db(conn)
        conn.execute("TRUNCATE chunks")
        conn.commit()
        ingest(conn, "lab-notes", SAMPLE)

        local_tools = build_local_tools(conn)

        prompts = [
            # KB answers -> search_knowledge_base only, no web_search needed
            "How do I make Kyverno work with Cosign v3?",
            # KB can't answer, REAL web search should fire this time
            "What is the latest stable Kubernetes version?",
            # no search needed
            "What is 47 plus 55?",
            "Latest news swedish elections, gist?",
        ]
        for p in prompts:
            print(f"\nUSER: {p}")
            print(f"ASSISTANT: {run(p, local_tools)}")
