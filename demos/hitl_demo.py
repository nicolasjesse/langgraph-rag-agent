"""Force the verifier to reject twice → interrupt → simulate human approval.

This mirrors what would happen in real CLI use: the user sees a paused
graph, types their choice, and the graph resumes. Here we feed the
choice programmatically so the demo can run unattended.

Run: python demos/hitl_demo.py [approve|reject|rewrite]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent

attempts = {"count": 0}
real_planner = agent.planner_node

BAD_ANSWERS = [
    # Attempt 1: hallucinate the company.
    "LangGraph was created by Microsoft Research in 2018 [microsoft.md].",
    # Attempt 2: hallucinate something else, so the verifier rejects again.
    "LangGraph is a Rust-only framework that requires a paid IBM Cloud "
    "subscription to use [ibm_langgraph.md].",
]


def fail_twice_planner(state):
    attempts["count"] += 1
    if attempts["count"] <= len(BAD_ANSWERS):
        return {"answer": BAD_ANSWERS[attempts["count"] - 1]}
    return real_planner(state)


def _print_updates(chunk):
    for node_name, update in chunk.items():
        if node_name == "supervisor":
            print(f"[supervisor: {update.get('category')}]")
        elif node_name == "retrieval":
            docs = update.get("retrieved_docs", [])
            sources = sorted({d["source"] for d in docs})
            print(f"[retrieval: {len(docs)} chunks from {len(sources)} source(s)]")
        elif node_name == "planner":
            print(f"\n[planner attempt #{attempts['count']}]")
            answer = update.get("answer", "")
            print(answer if len(answer) < 500 else answer[:500] + "...")
        elif node_name == "verifier":
            v = update
            tag = "PASS" if v.get("verifier_passes") else "REJECT"
            print(f"\n[verifier: {tag} @ iter {v.get('iteration_count')}] {v.get('verifier_feedback')}")


def main() -> int:
    load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")):
        print("ERROR: API keys missing", file=sys.stderr)
        return 1

    action = (sys.argv[1] if len(sys.argv) > 1 else "approve").lower()
    if action not in ("approve", "reject", "rewrite"):
        print(f"unknown action: {action!r}; expected one of: approve | reject | rewrite", file=sys.stderr)
        return 2

    print("Q: What is LangGraph?")
    print("(planner is patched to deliberately hallucinate twice; HITL kicks in at iter=2)")
    print(f"(simulated human action on interrupt: {action!r})\n")

    decision = {"action": action}
    if action == "rewrite":
        decision["answer"] = "Human override: LangGraph is what you say it is."

    with patch.object(agent, "planner_node", fail_twice_planner):
        checkpointer = InMemorySaver()
        graph = agent.build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "hitl-demo"}}
        next_input = {"query": "What is LangGraph?"}

        while True:
            interrupted = False
            for chunk in graph.stream(next_input, config=config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    payload = chunk["__interrupt__"][0].value
                    print()
                    print("=" * 60)
                    print("[!] HUMAN REVIEW REQUIRED  (simulated)")
                    print("=" * 60)
                    print(f"After {payload['iteration']} attempts, verifier still rejects.")
                    print(f"Verifier reason: {payload['verifier_reason']}")
                    print(f"-> simulated decision: {decision}")
                    next_input = Command(resume=decision)
                    interrupted = True
                    break
                _print_updates(chunk)
            if not interrupted:
                break

        final = graph.get_state(config).values
        print("\n--- final state ---")
        print(f"answer:           {final['answer'][:200]}...")
        print(f"verifier_passes:  {final.get('verifier_passes')}")
        print(f"verifier_feedback: {final.get('verifier_feedback')}")
        print(f"iteration_count:  {final.get('iteration_count')}")

    print(f"\nTotal planner attempts: {attempts['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
