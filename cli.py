"""CLI driver for the LangGraph RAG agent.

This is one of two adapters around agent.py — it knows how to:
  - parse argv,
  - open a SqliteSaver for checkpoint persistence,
  - stream graph events to stdout (token-by-token + node traces),
  - prompt the user when the graph hits an interrupt() and resume.

The agent module itself doesn't know any of this. A future
lambda_handler.py is the second adapter — same build_graph(), different
driver.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import MAX_ITERATIONS, build_graph

# Langfuse is optional — only attach if keys are set in .env.
try:
    from langfuse.langchain import CallbackHandler as _LangfuseHandler
except ImportError:  # pragma: no cover
    _LangfuseHandler = None

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DB = CHECKPOINT_DIR / "agent.db"


def _print_chunk_content(content) -> None:
    """Print a streamed message chunk, handling both str and block-list shapes."""
    if isinstance(content, str):
        print(content, end="", flush=True)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""), end="", flush=True)


def _handle_interrupt(interrupt_objs) -> dict:
    """Render the interrupt context and ask the user what to do."""
    payload = interrupt_objs[0].value
    print()
    print("=" * 60)
    print("[!] HUMAN REVIEW REQUIRED")
    print("=" * 60)
    print(f"After {payload['iteration']} attempt(s), the verifier still rejected the answer.")
    print(f"\nVerifier's reason:\n  {payload['verifier_reason']}")
    print(f"\nCurrent answer:\n{payload['answer']}")
    print()
    print("Options:")
    print("  [a]pprove — accept this answer as-is")
    print("  [r]eject  — give up, return current answer flagged as rejected")
    print("  [w]rite   — type a custom answer to use instead")
    choice = input("\nYour choice [a/r/w]: ").strip().lower()

    if choice in ("a", "approve"):
        return {"action": "approve"}
    if choice in ("w", "write", "rewrite"):
        custom = input("Type your answer:\n> ").strip()
        return {"action": "rewrite", "answer": custom}
    return {"action": "reject"}


def _build_callbacks() -> list:
    """Attach a Langfuse callback handler if its keys are present in .env."""
    if _LangfuseHandler is None:
        return []
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    return [_LangfuseHandler()]


def _stream_to_stdout(graph, query: str, thread_id: str, callbacks: list | None = None) -> None:
    """Stream graph events to stdout, handling interrupts by prompting the user."""
    print(f"Q: {query}\n")
    answer_started = False
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks or [],
        "metadata": {"langfuse_session_id": thread_id},
    }
    next_input = {"query": query}

    while True:
        interrupted = False
        for mode, chunk in graph.stream(
            next_input,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                if "__interrupt__" in chunk:
                    decision = _handle_interrupt(chunk["__interrupt__"])
                    next_input = Command(resume=decision)
                    interrupted = True
                    answer_started = False
                    break
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
                        print(f"\n[verifier: pass — {v.get('verifier_feedback', '')}]")
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

        if not interrupted:
            break

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

    callbacks = _build_callbacks()
    if callbacks:
        print("[langfuse: tracing on]\n")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    try:
        with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            _stream_to_stdout(graph, query, args.thread, callbacks=callbacks)
    finally:
        # Ensure pending Langfuse events are sent before the process exits.
        if callbacks:
            from langfuse import get_client
            try:
                get_client().flush()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
