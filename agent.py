"""Day 2 agent with supervisor + verifier loop.

Graph:
    START -> supervisor -> {direct_answer -> END
                          | retrieval -> planner -> verifier -> {END | retry-to-planner}}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DB = CHECKPOINT_DIR / "agent.db"
COLLECTION_NAME = "langchain_docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5"           # planner — heavy reasoning
CLASSIFIER_MODEL = "claude-haiku-4-5"      # supervisor + verifier + direct_answer — cheap
TOP_K = 5
MAX_ITERATIONS = 3                          # planner/verifier loop budget

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

VERIFIER_PROMPT = """You are a quality checker for a RAG system.

You will be given:
1. The user's question
2. The documentation chunks that were retrieved
3. An answer that was generated

Decide if the answer should pass:
- It must use ONLY information present in the provided docs (no hallucinated facts).
- It must actually address the question.
- Minor stylistic or formatting issues are fine — only fail for hallucinations,
  factual errors, or off-topic responses.

Return passes=true if it passes; otherwise passes=false with a one-sentence reason
that the planner can use to correct the next attempt.
"""


class Classification(BaseModel):
    """Schema the supervisor LLM is forced to return."""

    category: Literal["simple", "needs_retrieval", "escalate"] = Field(
        description="The route this query should take."
    )
    reason: str = Field(description="One sentence explaining the classification.")


class Verification(BaseModel):
    """Schema the verifier LLM is forced to return."""

    passes: bool = Field(description="True if the answer is well-grounded in the docs.")
    reason: str = Field(description="One sentence explaining the decision.")


class AgentState(TypedDict):
    query: str
    category: str
    retrieved_docs: list[dict]
    answer: str
    iteration_count: int
    verifier_passes: bool
    verifier_feedback: str


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

    feedback = state.get("verifier_feedback") or ""
    iteration = state.get("iteration_count", 0)
    if iteration > 0 and feedback:
        user_message += (
            f"\n\nNote: a previous attempt was rejected by the verifier. "
            f"Reason: {feedback}\nProduce a corrected answer that addresses this."
        )

    response = llm.invoke(
        [
            ("system", PLANNER_PROMPT),
            ("user", user_message),
        ]
    )
    return {"answer": response.content}


def verifier_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model=CLASSIFIER_MODEL, temperature=0).with_structured_output(
        Verification
    )
    context = "\n\n".join(
        f"[{d['source']}]\n{d['text']}" for d in state["retrieved_docs"]
    )
    user_message = (
        f"Question: {state['query']}\n\n"
        f"Retrieved docs:\n{context}\n\n"
        f"Answer to verify:\n{state['answer']}"
    )
    result = llm.invoke(
        [
            ("system", VERIFIER_PROMPT),
            ("user", user_message),
        ]
    )
    return {
        "verifier_passes": result.passes,
        "verifier_feedback": result.reason,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


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


def _route_after_verifier(state: AgentState) -> str:
    """Conditional edge after verifier — pass, retry, or give up."""
    if state.get("verifier_passes"):
        return "end"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"   # budget exhausted — return what we have
    return "retry"


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
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
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"end": END, "retry": "planner"},
    )
    graph.add_edge("direct_answer", END)

    return graph.compile(checkpointer=checkpointer)


def _print_chunk_content(content) -> None:
    """Print a streamed message chunk's content, handling both str and block-list shapes."""
    if isinstance(content, str):
        print(content, end="", flush=True)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""), end="", flush=True)


def _stream_to_stdout(graph, query: str, thread_id: str) -> None:
    """Stream graph events to stdout — supervisor/retrieval/verifier traces + answer tokens."""
    print(f"Q: {query}\n")
    answer_started = False
    config = {"configurable": {"thread_id": thread_id}}

    for mode, chunk in graph.stream(
        {"query": query},
        config=config,
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
            elif "verifier" in chunk:
                v = chunk["verifier"]
                if v.get("verifier_passes"):
                    print(f"\n[verifier: pass]")
                else:
                    n = v.get("iteration_count", "?")
                    reason = v.get("verifier_feedback", "")
                    if n != "?" and n >= MAX_ITERATIONS:
                        print(f"\n[verifier: fail (budget exhausted at iter {n}) — {reason}]")
                    else:
                        print(f"\n[verifier: retry (iter {n}) — {reason}]\n")
                        answer_started = False
        elif mode == "messages":
            msg_chunk, metadata = chunk
            node = (metadata or {}).get("langgraph_node")
            if node not in ("planner", "direct_answer"):
                continue
            if not answer_started:
                print("A: ", end="", flush=True)
                answer_started = True
            _print_chunk_content(msg_chunk.content)

    print(f"\n[checkpointed: thread={thread_id}]")


def main() -> int:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="LangGraph RAG agent CLI.")
    parser.add_argument("query", nargs="+", help="Your question.")
    parser.add_argument(
        "--thread", "-t", default="default",
        help="Thread ID for checkpoint persistence (default: 'default').",
    )
    args = parser.parse_args()
    query = " ".join(args.query)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        _stream_to_stdout(graph, query, args.thread)

    return 0


if __name__ == "__main__":
    sys.exit(main())
