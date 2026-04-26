"""Day 1 agent: minimal LangGraph that retrieves from Chroma and answers via Claude.

Graph: START -> retrieval -> planner -> END
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "langchain_docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5"
TOP_K = 5

SYSTEM_PROMPT = """You are an AI assistant for the LangChain and LangGraph Python documentation.

Answer the user's question using ONLY the provided context. If the context does not
contain enough information to answer, say so clearly — do not invent details.

Cite sources by appending the filename in brackets after each fact, e.g.
[langgraph_overview.md]. Multiple citations are fine.
"""


class AgentState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    answer: str


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )
    return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)


def retrieval_node(state: AgentState) -> dict:
    collection = _get_collection()
    results = collection.query(query_texts=[state["query"]], n_results=TOP_K)
    docs = [
        {"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {"retrieved_docs": docs}


def planner_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)
    context = "\n\n".join(
        f"[{d['source']}]\n{d['text']}" for d in state["retrieved_docs"]
    )
    user_message = f"Context:\n\n{context}\n\nQuestion: {state['query']}"
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", user_message),
        ]
    )
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "planner")
    graph.add_edge("planner", END)
    return graph.compile()


def main() -> int:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env", file=sys.stderr)
        return 1

    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question here"', file=sys.stderr)
        return 1
    query = " ".join(sys.argv[1:])

    graph = build_graph()
    result = graph.invoke({"query": query})
    print(result["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
