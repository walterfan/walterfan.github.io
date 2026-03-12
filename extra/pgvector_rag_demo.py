#!/usr/bin/env python3
"""
pgvector RAG Demo — Read markdown files, embed, store in PostgreSQL, then query.

Reads configuration from the project .env file:
  - DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASS (or DATABASE_URL)
  - EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL
  - LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

Usage:
  # Index markdown files from a directory:
  python pgvector_rag_demo.py index ./path/to/markdown/dir

  # Query the knowledge base:
  python pgvector_rag_demo.py query "How does the RAG engine work?"

  # Interactive mode:
  python pgvector_rag_demo.py chat

  # Show stats:
  python pgvector_rag_demo.py stats

Prerequisites:
  pip install psycopg2-binary openai python-dotenv
  docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 ankane/pgvector
"""

import os
import sys
import glob
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Load project root .env first, then local .env (local values take priority)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(override=True)


# ============================================================
# Configuration from .env
# ============================================================

def get_db_config() -> dict:
    """Build psycopg2 connection kwargs from env vars."""
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgresql"):
        # Parse DATABASE_URL: postgresql://user:pass@host:port/dbname
        from urllib.parse import urlparse
        p = urlparse(db_url)
        return {
            "host": p.hostname or "localhost",
            "port": p.port or 5432,
            "dbname": p.path.lstrip("/") or "rag_db",
            "user": p.username or "postgres",
            "password": p.password or "postgres",
        }
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "rag_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASS", "postgres"),
    }


EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")
LLM_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "true").lower() not in ("false", "0", "no")

TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

print(f"EMBEDDING_BASE_URL: {EMBEDDING_BASE_URL}")
print(f"EMBEDDING_API_KEY: {EMBEDDING_API_KEY[:5]}...")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"LLM_BASE_URL: {LLM_BASE_URL}")
print(f"LLM_API_KEY: {LLM_API_KEY[:5]}...")
print(f"LLM_MODEL: {LLM_MODEL}")
print(f"LLM_VERIFY_SSL: {LLM_VERIFY_SSL}")

# ============================================================
# Embedding via OpenAI-compatible API
# ============================================================

def get_embedding_client():
    import httpx
    from openai import OpenAI

    extra_headers = {"X-Api-Key": EMBEDDING_API_KEY}
    http_client = httpx.Client(
        headers=extra_headers,
        verify=LLM_VERIFY_SSL,
    )

    return OpenAI(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        http_client=http_client,
    )


def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = get_embedding_client()
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


# ============================================================
# Text chunking (sentence-aware)
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "。", "! ", "? "]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


# ============================================================
# Markdown file reading
# ============================================================

def read_markdown_files(directory: str) -> list[dict]:
    """Read all .md files from a directory, return list of {title, content, source}."""
    docs = []
    md_patterns = [os.path.join(directory, "*.md"), os.path.join(directory, "**/*.md")]

    seen = set()
    for pattern in md_patterns:
        for filepath in glob.glob(pattern, recursive=True):
            filepath = os.path.abspath(filepath)
            if filepath in seen:
                continue
            seen.add(filepath)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"  ⚠ Skipping {filepath}: {e}")
                continue

            if not content.strip():
                continue

            # Extract title from first heading or filename
            title = Path(filepath).stem
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
                if line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip()
                    break

            docs.append({
                "title": title,
                "content": content,
                "source": filepath,
            })

    return docs


# ============================================================
# Database operations
# ============================================================

def get_connection():
    import psycopg2
    cfg = get_db_config()
    conn = psycopg2.connect(**cfg)
    print(f"✅ Connected to PostgreSQL: {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    return conn


def _detect_embedding_dim() -> int:
    """Probe the embedding model with a short text to discover its output dimension."""
    try:
        emb = get_embedding("dimension probe")
        dim = len(emb)
        print(f"✅ Detected embedding dimension: {dim} (model: {EMBEDDING_MODEL})")
        return dim
    except Exception as e:
        print(f"⚠ Could not probe embedding dimension ({e}), defaulting to 1536")
        return 1536


def init_db(conn):
    """Create pgvector extension and documents table with correct vector dimension."""
    dim = _detect_embedding_dim()

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Check if table already exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'rag_documents'
            );
        """)
        table_exists = cur.fetchone()[0]

        if not table_exists:
            cur.execute(f"""
                CREATE TABLE rag_documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT '',
                    chunk_index INTEGER DEFAULT 0,
                    embedding vector({dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute(f"""
                CREATE INDEX idx_rag_docs_embedding
                ON rag_documents
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            print(f"✅ Created table rag_documents with vector({dim}) + HNSW index")
        else:
            # Verify existing column dimension matches
            cur.execute("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'rag_documents'::regclass
                  AND attname = 'embedding';
            """)
            row = cur.fetchone()
            existing_dim = row[0] if row and row[0] > 0 else None
            if existing_dim and existing_dim != dim:
                print(f"⚠ Warning: existing embedding column is vector({existing_dim}), "
                      f"but current model produces {dim}-dim vectors. "
                      f"Run 'reset' and re-index if you switched models.")
            print(f"✅ Table rag_documents already exists (vector dim: {existing_dim or 'unset'})")

        conn.commit()


def index_documents(conn, docs: list[dict]):
    """Chunk documents, compute embeddings, insert into pgvector."""
    from psycopg2.extras import execute_values

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc["content"])
        if not chunks:
            continue

        print(f"  📄 {doc['title']}: {len(chunks)} chunks", end="")

        # Batch embed (OpenAI API supports up to 2048 texts)
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            all_embeddings.extend(get_embeddings(batch))
            print(".", end="", flush=True)
        print()

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            rows.append((doc["title"], chunk, doc.get("source", ""), i, emb))

        with conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO rag_documents (title, content, source, chunk_index, embedding)
                   VALUES %s""",
                rows,
                template="(%s, %s, %s, %s, %s::vector)",
            )
        conn.commit()
        total_chunks += len(rows)

    print(f"✅ Indexed {total_chunks} chunks from {len(docs)} documents")


def search_similar(conn, query: str, top_k: int = TOP_K) -> list[dict]:
    """Cosine similarity search."""
    query_emb = get_embedding(query)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, source, chunk_index,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_emb, query_emb, top_k),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_documents;")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT title) FROM rag_documents;")
        docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT source) FROM rag_documents;")
        sources = cur.fetchone()[0]
    return {"total_chunks": total, "documents": docs, "sources": sources}


# ============================================================
# RAG: Retrieve + Generate
# ============================================================

def rag_query(conn, question: str) -> str:
    """Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM."""
    import httpx
    from openai import OpenAI

    results = search_similar(conn, question)

    print(f"\n{'─' * 60}")
    print(f"🔍 Question: {question}")
    print(f"{'─' * 60}")
    print(f"📚 Retrieved {len(results)} chunks:")
    for i, r in enumerate(results, 1):
        sim = r["similarity"]
        print(f"  [{i}] {r['title']} (chunk #{r['chunk_index']}, similarity: {sim:.4f})")

    # Build context from retrieved chunks
    context_parts = []
    for r in results:
        snippet = r["content"][:500]
        context_parts.append(f"### {r['title']}\n{snippet}\n(source: {r['source']})")
    context = "\n\n".join(context_parts)

    # Call LLM
    http_client = httpx.Client(
        headers={"X-Api-Key": LLM_API_KEY},
        verify=LLM_VERIFY_SSL,
    )
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, http_client=http_client)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. Answer the user's question based on "
                "the provided reference materials. If the materials are insufficient, say so. "
                "Cite sources when possible. Answer in the same language as the question."
            ),
        },
        {
            "role": "user",
            "content": f"Reference materials:\n{context}\n\nQuestion: {question}",
        },
    ]

    print(f"\n📨 Messages to LLM ({LLM_MODEL}):")
    for msg in messages:
        role = msg["role"].upper()
        print(f"  [{role}]\n{textwrap.indent(msg['content'], '    ')}")
    print(f"{'─' * 60}")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    print(f"\n💬 Answer:\n{textwrap.fill(answer, width=80)}")
    return answer


# ============================================================
# CLI
# ============================================================

def print_usage():
    print("""
pgvector RAG Demo
=================

Usage:
  python pgvector_rag_demo.py index <directory>   Index markdown files from directory
  python pgvector_rag_demo.py query "<question>"   One-shot RAG query
  python pgvector_rag_demo.py chat                 Interactive RAG chat
  python pgvector_rag_demo.py stats                Show knowledge base stats
  python pgvector_rag_demo.py reset                Clear all indexed data

Environment (from .env):
  DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASS
  EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    print(f"📋 Config:")
    print(f"   Embedding: {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")
    print(f"   LLM:       {LLM_MODEL} @ {LLM_BASE_URL}")
    db_cfg = get_db_config()
    print(f"   Database:  {db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}")
    print()

    conn = get_connection()
    try:
        init_db(conn)

        if command == "index":
            if len(sys.argv) < 3:
                print("Error: provide a directory path")
                print("  python pgvector_rag_demo.py index ./path/to/markdown/dir")
                sys.exit(1)

            directory = sys.argv[2]
            if not os.path.isdir(directory):
                print(f"Error: '{directory}' is not a directory")
                sys.exit(1)

            print(f"\n📂 Scanning markdown files in: {directory}")
            docs = read_markdown_files(directory)
            if not docs:
                print("No markdown files found.")
                sys.exit(0)

            print(f"Found {len(docs)} markdown files:\n")
            for d in docs:
                chars = len(d["content"])
                print(f"  • {d['title']} ({chars:,} chars) — {d['source']}")
            print()

            index_documents(conn, docs)

        elif command == "query":
            if len(sys.argv) < 3:
                print("Error: provide a question")
                print('  python pgvector_rag_demo.py query "What is pgvector?"')
                sys.exit(1)

            question = " ".join(sys.argv[2:])
            rag_query(conn, question)

        elif command == "chat":
            print("\n🤖 Interactive RAG chat (type 'quit' to exit)\n")
            while True:
                try:
                    question = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break

                if not question or question.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break

                if question.lower() == "stats":
                    s = get_stats(conn)
                    print(f"  📊 {s['total_chunks']} chunks, {s['documents']} documents, {s['sources']} sources")
                    continue

                rag_query(conn, question)
                print()

        elif command == "stats":
            s = get_stats(conn)
            print(f"\n📊 Knowledge Base Stats:")
            print(f"   Total chunks:  {s['total_chunks']}")
            print(f"   Documents:     {s['documents']}")
            print(f"   Sources:       {s['sources']}")

        elif command == "reset":
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS rag_documents;")
                conn.commit()
            print("🗑️  All indexed data cleared. Run 'index' to re-index.")

        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)

    finally:
        conn.close()
        print("\n✅ Connection closed")


if __name__ == "__main__":
    main()
