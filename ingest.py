"""Ingest corpus/ markdown files into a local ChromaDB collection.

Run once after scrape_corpus.py. Re-running drops and rebuilds the collection.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "langchain_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def load_corpus() -> list[tuple[str, str]]:
    """Return [(filename, full_text)] for every .md file in corpus/."""
    files = sorted(CORPUS_DIR.glob("*.md"))
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def chunk_corpus(docs: list[tuple[str, str]]) -> tuple[list[str], list[dict], list[str]]:
    """Split each doc into overlapping chunks. Returns (texts, metadatas, ids)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,        # tokens → chars (~4 chars per token)
        chunk_overlap=CHUNK_OVERLAP * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts, metadatas, ids = [], [], []
    for filename, content in docs:
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}::chunk_{i}")
    return texts, metadatas, ids


def main() -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env.", file=sys.stderr)
        return 1

    print(f"Loading corpus from {CORPUS_DIR}/")
    docs = load_corpus()
    print(f"  {len(docs)} markdown files")

    print(f"Chunking with size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")
    texts, metadatas, ids = chunk_corpus(docs)
    print(f"  {len(texts)} chunks total")

    print(f"Initializing ChromaDB at {CHROMA_DIR}/")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop the collection if it exists, so re-runs are clean.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  dropped existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  created collection: {COLLECTION_NAME}")

    print(f"Embedding & adding chunks in batches of {BATCH_SIZE}...")
    start = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(texts))
        collection.add(
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )
        print(f"  added {batch_end}/{len(texts)} chunks")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Collection has {collection.count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
