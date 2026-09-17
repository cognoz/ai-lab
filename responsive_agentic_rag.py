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

from rag import init_db, retrieve, connect_with_retry
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
        # Tool availability is STAGED by how far we are into the fallback:
        #
        #   Stage 1 (KB not yet exhausted): offer everything. The model
        #     picks the right tool for the question — search_knowledge_base
        #     for ops questions, get_weather / add for their own kinds.
        #
        #   Stage 2 (KB exhausted, kb_calls_made >= max_kb_calls): the
        #     question reached here because the KB couldn't answer it, so
        #     it's an ops/knowledge question that now needs the web. Offer
        #     ONLY web_search. We remove get_weather and add too — not just
        #     the KB — because otherwise the model thrashes on the nearest
        #     available wrong tool (we watched it jam "Kubernetes" into
        #     get_weather's city field eight times). If the only escalation
        #     tool is web_search, that thrashing is impossible: the sole
        #     option left is the correct one.
        if kb_calls_made >= max_kb_calls:
            available_tools = [t for t in TOOLS if t.get("type") == "web_search"]
        else:
            available_tools = TOOLS

        # THE FIX for "the model answered from memory without searching":
        # on turn 0 only, force tool_choice="required" — the model MUST
        # call SOME tool, it just can't skip straight to a free-text
        # answer. We do NOT force it to call search_knowledge_base
        # specifically (that would wrongly force a KB search on "what's
        # 47+55?" too) — "required" lets it still pick the RIGHT tool
        # (get_weather, add, or search_knowledge_base), it just can't
        # pick NO tool. After turn 0, back to "auto": once results are
        # in, the model should be free to decide it has enough to answer.
        tool_choice = "required" if _turn == 0 else "auto"

        resp = client.responses.create(
            model=MODEL, input=input_items, tools=available_tools, tool_choice=tool_choice
        )
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
        local_tools = build_local_tools(conn)
        # NOTE: this file no longer ingests here — it now expects you've
        # already run `python3 ingest_k8s_docs.py` once to populate the
        # real Kubernetes-docs corpus. Re-ingesting the 3-sentence toy
        # SAMPLE on every run would overwrite it. If you want the toy
        # corpus back for a quick sanity check, ingest it explicitly with
        # a one-off script instead of doing it here.

        prompts = [
            "How do I expose my app so users outside the cluster can reach it?",
            "What's the difference between a Deployment and a StatefulSet?",
            "What is the latest stable Kubernetes version?",  # KB miss -> web
            "What is 47 plus 55?",
        ]
        for p in prompts:
            print(f"\nUSER: {p}")
            print(f"ASSISTANT: {run(p, local_tools)}")
