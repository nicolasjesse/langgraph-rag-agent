"""Force the verifier loop to reject and retry — proves the recovery path works.

We monkey-patch planner_node so the first attempt returns a deliberately
hallucinated answer. The real verifier (Haiku) should reject it, the loop
routes back to the real planner, and the second attempt streams a grounded
answer that passes.

Run: python demos/verifier_loop_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Make agent.py importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

import agent

attempts = {"count": 0}
real_planner = agent.planner_node

BAD_ANSWER = (
    "LangGraph was created by Microsoft Research in 2018 as a JavaScript "
    "framework for enterprise agents. It requires a paid Azure subscription "
    "and only supports OpenAI's GPT-4 model. [microsoft_langgraph.md]"
)


def fail_once_planner(state):
    """First call: hallucinate. Subsequent calls: delegate to the real planner."""
    attempts["count"] += 1
    if attempts["count"] == 1:
        return {"answer": BAD_ANSWER}
    return real_planner(state)


def main() -> int:
    load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env", file=sys.stderr)
        return 1

    query = "What is LangGraph?"
    print(f"Q: {query}")
    print("(planner is patched to deliberately hallucinate on attempt #1)\n")

    with patch.object(agent, "planner_node", fail_once_planner):
        checkpointer = InMemorySaver()
        graph = agent.build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "verifier-loop-demo"}}

        for chunk in graph.stream(
            {"query": query},
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name == "supervisor":
                    print(f"[supervisor: {update.get('category')}]")
                elif node_name == "retrieval":
                    docs = update.get("retrieved_docs", [])
                    sources = sorted({d["source"] for d in docs})
                    print(f"[retrieval: {len(docs)} chunks from {len(sources)} source(s)]")
                elif node_name == "planner":
                    label = (
                        "patched (hallucinated)"
                        if attempts["count"] == 1
                        else "real (after verifier feedback)"
                    )
                    print(f"\n[planner attempt #{attempts['count']} — {label}]")
                    answer = update.get("answer", "")
                    preview = answer if len(answer) < 600 else answer[:600] + "..."
                    print(preview)
                elif node_name == "verifier":
                    v = update
                    if v.get("verifier_passes"):
                        print(f"\n[verifier: PASS @ iter {v.get('iteration_count')}] {v.get('verifier_feedback')}")
                    else:
                        print(f"\n[verifier: REJECT @ iter {v.get('iteration_count')}] {v.get('verifier_feedback')}")

    print(f"\nTotal planner attempts: {attempts['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
