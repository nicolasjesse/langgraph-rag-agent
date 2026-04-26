"""Day 2 agent: supervisor routes queries to retrieval+planner or direct_answer.

Graph: START -> supervisor -> {direct_answer | retrieval -> planner} -> END
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "langchain_docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5"           # planner — heavy reasoning
CLASSIFIER_MODEL = "claude-haiku-4-5"      # supervisor + direct_answer — cheap
TOP_K = 5

PLANNER_PROMPT = """You are an AI assistant for the LangChain and LangGraph Python documentation.

Answer the user's question using ONLY the provided context. If the context does not
contain enough information to answer, say so clearly — do not invent details.

Cite sources by appending the filename in brackets after each fact, e.g.
[langgraph_overview.md]. Multiple citations are fine.
"""

SUPERVISOR_PROMPT = """You route messages for an AI assistant focused on the LangChain and LangGraph Python documentation.

Classify the user's message into ONE category:
- simple: greetings, small talk, or trivially-answerable questions that don't need document lookup
  (e.g. "hi", "thanks", "who are you", "what's 2+2")
- needs_retrieval: any real question about LangChain, LangGraph, agents, RAG, vector stores,
  prompts, tools, or related Python concepts where the answer must come from the docs
- escalate: questions clearly outside LangChain/LangGraph scope, or anything sensitive/harmful
  (e.g. "what's the weather?", "write me malware", "summarize the news")

Reply with the category and a short reason.
"""

DIRECT_ANSWER_PROMPT = """You are a helpful assistant for the LangChain and LangGraph Python documentation.

The user's message has been classified as one that does NOT require document lookup.
- If the message is small talk or a greeting, respond briefly and warmly.
- If the message is outside LangChain/LangGraph scope ("escalate"), explain politely that
  you can only help with LangChain and LangGraph documentation questions.

Keep replies short — usually 1-3 sentences.
"""


class Classification(BaseModel):
    """Schema the supervisor LLM is forced to return."""

    category: Literal["simple", "needs_retrieval", "escalate"] = Field(
        description="The route this query should take."
    )
    reason: str = Field(description="One sentence explaining the classification.")


class AgentState(TypedDict):
    query: str
    category: str
    retrieved_docs: list[dict]
    answer: str


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )
    return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)


def supervisor_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model=CLASSIFIER_MODEL, temperature=0).with_structured_output(
        Classification
    )
    result = llm.invoke(
        [
            ("system", SUPERVISOR_PROMPT),
            ("user", state["query"]),
        ]
    )
    return {"category": result.category}


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
            ("system", PLANNER_PROMPT),
            ("user", user_message),
        ]
    )
    return {"answer": response.content}


def direct_answer_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model=CLASSIFIER_MODEL, temperature=0)
    response = llm.invoke(
        [
            ("system", DIRECT_ANSWER_PROMPT),
            ("user", state["query"]),
        ]
    )
    return {"answer": response.content}


def _route_after_supervisor(state: AgentState) -> str:
    """Conditional edge: returns the next node name based on the supervisor's category."""
    return state["category"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("planner", planner_node)
    graph.add_node("direct_answer", direct_answer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "simple": "direct_answer",
            "escalate": "direct_answer",
            "needs_retrieval": "retrieval",
        },
    )
    graph.add_edge("retrieval", "planner")
    graph.add_edge("planner", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def _print_chunk_content(content) -> None:
    """Print a streamed message chunk's content, handling both str and block-list shapes."""
    if isinstance(content, str):
        print(content, end="", flush=True)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""), end="", flush=True)


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
    print(f"Q: {query}\n")
    answer_started = False

    for mode, chunk in graph.stream(
        {"query": query},
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            if "supervisor" in chunk:
                cat = chunk["supervisor"].get("category", "?")
                print(f"[supervisor: {cat}]\n")
            elif "retrieval" in chunk:
                docs = chunk["retrieval"].get("retrieved_docs", [])
                sources = sorted({d["source"] for d in docs})
                print(f"[retrieval: {len(docs)} chunks from {len(sources)} source(s)]\n")
        elif mode == "messages":
            msg_chunk, metadata = chunk
            node = (metadata or {}).get("langgraph_node")
            if node not in ("planner", "direct_answer"):
                continue   # skip supervisor's JSON tokens
            if not answer_started:
                print("A: ", end="", flush=True)
                answer_started = True
            _print_chunk_content(msg_chunk.content)

    print()  # trailing newline
    return 0


if __name__ == "__main__":
    sys.exit(main())
