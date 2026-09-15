# AI agentic lab
Main goal of the lab - to grasp all main concepts behind RAG, agents etc.
Here we build our agentic system from scratch, making one step in time.
We don't use frameworks from the start in order to get more deeper understanding
of the all machinery.

## rag.py
In this file we implemented simple rag pipeline, with simple deterministic flow. Main concepts tried out -
splitting info in chunks (both by symbols and paragraphs), embedding in vectors (using simple embed model) and ingesting it in Postgres
retrieval of the context from Postgres (ranked by semantic similarity via cosine distance between question and chunks)
and sending that context to simple OpenAI chat model asking to use only context for the answer). No loops, no
extra tools, just storing/retrieving/feeding the context to OpenAI chat model. Retrieval always runs, on every query,
because the pipeline is fixed — the model has no say in whether to search

## agent\_loop.py
Here we add both tooling and loops. We don't search for actual data through the API, just using a stub for data - because main point of
this excercise is to grasp the idea of agentic flow, flow there model running in loops and can freely switch between different tools without
hard-coded conditionals. So we supply tools definitions with schemas, prompts and let the model decide what to use. Loops running until model
stops asking for tools and returns a final answer.

## agentic\_rag.py
Here we combine rag and agentic loops/tools basically. So both local database read from rag.py and tools from agent\_loop.py are present.
We reuse code from both scripts as functions, so no duplicated code. On top of that, we added web\_search tool, which currently(Phase 1) just use
predefined context for answers (we are more interested in orchestration layer right now, to see how model decides to fallback to the other tool(web search)
but in the next Phase the script will utilize modern Responses API from OpenAI to perform web search. 

Main difference between RAG pipeline and agentic flow is that with RAG we control when and how to retrieve the data, but with agentic flow we let model to decide
it itself - when to run/stop/which tool to use.
