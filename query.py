"""Quick CLI for inspecting retrieval against the local Chroma collection.

Usage:
    python query.py "your question here"
    python query.py "your question" -k 5

This is a developer tool — not part of the agent. Use it to sanity-check
the corpus and tune chunking / k before wiring retrieval into the graph.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "langchain_docs"
EMBED_MODEL = "text-embedding-3-small"


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local Chroma collection.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of results (default 3)")
    parser.add_argument("--full", action="store_true", help="Print full chunk text instead of preview")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env.", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    results = collection.query(query_texts=[args.query], n_results=args.top_k)

    print(f"Query: {args.query}")
    print(f"Collection: {COLLECTION_NAME} ({collection.count()} chunks total)\n")

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        similarity = 1 - dist
        print(f"--- #{i}  similarity={similarity:.3f}  source={meta['source']}  chunk={meta['chunk_index']} ---")
        if args.full:
            print(doc)
        else:
            preview = doc.strip().replace("\n", " ")
            print(preview[:400] + ("…" if len(preview) > 400 else ""))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
