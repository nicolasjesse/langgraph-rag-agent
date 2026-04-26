"""LangGraph RAG agent — graph definition only.

This module is the *brain*: state schema, prompts, nodes, and the graph
wiring. It does not handle I/O, CLI, or persistence layer choices — those
live in cli.py (and a future lambda_handler.py for deployment).

Graph:
    START -> supervisor -> {direct_answer -> END
                          | retrieval -> planner -> verifier -> {END | retry-to-planner}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "langchain_docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5"           # planner — heavy reasoning
CLASSIFIER_MODEL = "claude-haiku-4-5"      # supervisor + verifier + direct_answer — cheap
TOP_K = 5
MAX_ITERATIONS = 3                          # planner/verifier loop budget
HITL_TRIGGER = 2                            # iteration at which to escalate to a human

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
    # NOTE: on resume after interrupt(), this node re-runs from the top — so the
    # LLM call below executes a second time. Cost is small (Haiku) and the
    # verifier is deterministic at temperature=0. A production split would
    # move the LLM call into a separate node so resume doesn't repeat it.
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

    new_iter = state.get("iteration_count", 0) + 1

    # Escalate to a human when the verifier rejects at the HITL trigger iteration.
    if not result.passes and new_iter == HITL_TRIGGER:
        decision = interrupt(
            {
                "type": "verifier_failed",
                "iteration": new_iter,
                "max_iterations": MAX_ITERATIONS,
                "verifier_reason": result.reason,
                "answer": state["answer"],
            }
        )
        action = (decision or {}).get("action")

        if action == "approve":
            return {
                "verifier_passes": True,
                "verifier_feedback": "approved by human",
                "iteration_count": new_iter,
            }
        if action == "rewrite":
            return {
                "answer": (decision or {}).get("answer", state["answer"]),
                "verifier_passes": True,
                "verifier_feedback": "rewritten by human",
                "iteration_count": new_iter,
            }
        # Anything else (including "reject") — give up. Bump iter to MAX so the
        # router takes the "end" path with verifier_passes=False.
        return {
            "verifier_passes": False,
            "verifier_feedback": f"rejected by human (verifier said: {result.reason})",
            "iteration_count": MAX_ITERATIONS,
        }

    return {
        "verifier_passes": result.passes,
        "verifier_feedback": result.reason,
        "iteration_count": new_iter,
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


