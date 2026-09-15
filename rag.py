"""
Phase 1b — A minimal RAG pipeline from scratch. No framework.

The whole pipeline, in order:
  ingest:  chunk docs -> embed each chunk -> store vectors in Postgres
  query:   embed the question -> retrieve nearest chunks -> stuff into
           the prompt -> ask the model to answer from them.

Vector store: Azure Database for PostgreSQL Flexible Server + pgvector.
Embeddings:   OpenAI text-embedding-3-small (1536 dims, cheap).

Run: OPENAI_API_KEY=sk-... DATABASE_URL=postgres://... python rag.py
"""
import os
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
CHAT_MODEL = "gpt-4o-mini"
DB_URL = os.environ["DATABASE_URL"]


# ---------------------------------------------------------------------------
# Chunking. Fixed-size character windows with overlap is the crude baseline;
# chunk_by_paragraph splits on structure, which gives sharper embeddings.
# ---------------------------------------------------------------------------
def chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def chunk_by_paragraph(text: str) -> list[str]:
    # Structure-aware: one logical note per chunk. Each chunk is one coherent
    # idea, so its embedding is sharp and retrieval pulls the right note.
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def embed(texts: list[str]) -> list[np.ndarray]:
    # Batch: one API call for many chunks. Returns one vector per input.
    # Return numpy arrays — pgvector's adapter accepts them directly.
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


# ---------------------------------------------------------------------------
# Schema. register_vector() on the connection installs the adapter that
# converts numpy arrays / lists <-> pgvector's `vector` type automatically.
# With it active, we pass vectors as plain %s params — NO ::vector casts,
# NO text literals. That is the correct, documented pgvector usage, and
# mixing in manual casts is what silently broke retrieval before.
# ---------------------------------------------------------------------------
def init_db(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()  # extension must be committed before the type OID exists
    register_vector(conn)  # fetches the vector OID, installs the adapter
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id     bigserial PRIMARY KEY,
            doc    text,
            body   text,
            embedding vector({EMBED_DIMS})
        )
        """
    )
    # No ANN index: on a small corpus pgvector does an exact brute-force scan,
    # which is both fast and perfectly accurate. Add HNSW only at scale.
    # Defensively drop any ANN index left over from an earlier run — a stale
    # IVFFlat index with probes=1 silently returns too few rows, which is a
    # nasty, hard-to-spot failure. TRUNCATE and CREATE TABLE IF NOT EXISTS
    # both leave an existing index in place, so we drop it explicitly.
    conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    conn.commit()


def ingest(conn, doc_name: str, text: str):
    chunks = chunk_by_paragraph(text)
    vectors = embed(chunks)
    with conn.cursor() as cur:
        for body, vec in zip(chunks, vectors):
            # vec is a numpy array; the registered adapter handles it.
            cur.execute(
                "INSERT INTO chunks (doc, body, embedding) VALUES (%s, %s, %s)",
                (doc_name, body, vec),
            )
    conn.commit()
    print(f"  ingested {len(chunks)} chunks from {doc_name!r}")


# ---------------------------------------------------------------------------
# Retrieval. Embed the question with the SAME model; `<=>` is pgvector's
# cosine-distance operator, ORDER BY it ASC = most similar first.
# ---------------------------------------------------------------------------
def retrieve(conn, question: str, k: int = 4) -> list[str]:
    qvec = embed([question])[0]  # numpy array; adapter handles it
    with conn.cursor() as cur:
        cur.execute(
            "SELECT body FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (qvec, k),
        )
        return [row[0] for row in cur.fetchall()]


def answer(conn, question: str) -> str:
    context = retrieve(conn, question)
    joined = "\n\n---\n\n".join(context)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{joined}\n\nQUESTION: {question}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


SAMPLE = """
Kyverno verifies image signatures at admission time. For Cosign v3
compatibility the policy must use type SigstoreBundle rather than the
older type Cosign, otherwise verification fails against bundles produced
by keyless signing.

Falco crashes on dual plugin registration. Setting config_files to an
empty list in the Falco config prevents the plugin being registered
twice, which resolves the startup crash.

Spot node pools require tolerations. Any workload scheduled onto a spot
pool must tolerate the taint with key kubernetes.azure.com/scalesetpriority,
value spot, effect NoSchedule.
"""

if __name__ == "__main__":
    with psycopg.connect(DB_URL, autocommit=False) as conn:
        init_db(conn)
        ingest(conn, "lab-notes", SAMPLE)
        for q in [
            "How do I make Kyverno work with Cosign v3?",
            "Why does Falco crash on startup?",
            "What database does the lab use?",  # not in context — should say so
        ]:
            print(f"\nQ: {q}\nA: {answer(conn, q)}")
