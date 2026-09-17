"""
Phase 1 extension — ingest a REAL corpus: official Kubernetes docs
(pulled from the kubernetes/website GitHub repo — the source of
kubernetes.io) instead of the 3-sentence toy SAMPLE.

This is where chunking strategy stops being academic. chunk_by_paragraph
(used for the toy SAMPLE) would produce hundreds of tiny, disconnected
chunks from a real doc page — every blank line is a new chunk, including
mid-explanation paragraph breaks that don't represent a topic change.

Real docs have STRUCTURE: markdown headers (##, ###) mark actual topic
boundaries. So this file adds chunk_by_markdown_headers(), which splits
each doc at its headers instead of blank lines — one chunk per section,
which is a real topic unit, not an arbitrary paragraph.

Also strips two kinds of non-content noise real docs carry that the toy
SAMPLE never had:
  - YAML frontmatter (---\ntitle: ...\n---) — metadata, not prose
  - Hugo shortcodes ({{< note >}}, {{< glossary_tooltip ... >}}) — this
    site's templating syntax, not something the model should read as text

Run: OPENAI_API_KEY=sk-... DATABASE_URL=postgres://... python ingest_k8s_docs.py
"""
import glob
import os
import re

from rag import connect_with_retry, init_db, embed

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_markdown(text: str) -> str:
    text = FRONTMATTER_RE.sub("", text)
    text = SHORTCODE_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    return text


def chunk_by_markdown_headers(text: str, max_chars: int = 1500) -> list[str]:
    """Split on ## / ### headers — each section is one topic. A section
    that's still too long (some K8s doc sections run very long) gets
    further split by character windows as a fallback, so no single chunk
    blows past what's useful to embed and retrieve as one unit."""
    # Split keeping the header with the section that follows it.
    parts = re.split(r"(?=^#{2,3} .+$)", text, flags=re.MULTILINE)
    sections = [p.strip() for p in parts if p.strip()]

    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # fallback: character windows with overlap, same idea as
            # rag.py's chunk(), for sections too long to embed as one unit
            step = max_chars - 150
            for i in range(0, len(section), step):
                piece = section[i:i + max_chars].strip()
                if piece:
                    chunks.append(piece)
    return chunks


def ingest_file(conn, path: str):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    text = clean_markdown(raw)
    doc_name = os.path.basename(path)
    chunks = chunk_by_markdown_headers(text)
    if not chunks:
        print(f"  skipped {doc_name} (no content after cleaning)")
        return 0

    vectors = embed(chunks)
    with conn.cursor() as cur:
        for body, vec in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO chunks (doc, body, embedding) VALUES (%s, %s, %s)",
                (doc_name, body, vec),
            )
    conn.commit()
    print(f"  ingested {len(chunks)} chunks from {doc_name}")
    return len(chunks)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    print(f"found {len(files)} docs in {CORPUS_DIR}")

    with connect_with_retry(autocommit=False) as conn:
        init_db(conn)
        conn.execute("TRUNCATE chunks")
        conn.commit()

        total = 0
        for path in files:
            total += ingest_file(conn, path)
        print(f"\ntotal chunks ingested: {total}")

