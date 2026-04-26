"""Mini eval harness — run a fixed test set, grade by keyword matching.

Each test case has a question + a list of expected keywords. We run the
question through the full graph (supervisor + retrieval + planner +
verifier), then check the final answer contains every expected keyword
(case-insensitive). Pass/fail per case, totals at the end.

Designed to be run as a CI gate: exits 0 only if every case passes.

Usage:
    python evals/run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make agent importable when running from repo root or evals/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent import build_graph

ROOT = Path(__file__).parent.parent
TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def grade(answer: str, expected_keywords: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, missing_keywords). All keywords must be present (case-insensitive)."""
    answer_lower = answer.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    return len(missing) == 0, missing


def run_one(question: str, case_id: str) -> dict:
    """Run a question through the graph, auto-rejecting on HITL so the run can complete."""
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"eval-{case_id}"}}
    next_input = {"query": question}

    while True:
        interrupted = False
        for chunk in graph.stream(next_input, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                next_input = Command(resume={"action": "reject"})
                interrupted = True
                break
        if not interrupted:
            break

    return graph.get_state(config).values


def main() -> int:
    load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env", file=sys.stderr)
        return 2

    test_set = json.loads(TEST_SET_PATH.read_text())
    print(f"Running {len(test_set)} eval cases\n")

    passed = 0
    failed_cases = []
    started = time.time()

    for tc in test_set:
        case_id = tc["id"]
        question = tc["question"]
        expected = tc["expected_keywords"]

        case_started = time.time()
        try:
            result = run_one(question, case_id)
            answer = result.get("answer", "") or ""
        except Exception as e:
            print(f"  ERR  {case_id}: {question[:60]}  ({type(e).__name__}: {e})")
            failed_cases.append({"id": case_id, "reason": f"exception: {e}"})
            continue
        case_elapsed = time.time() - case_started

        ok, missing = grade(answer, expected)
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {case_id}  ({case_elapsed:5.1f}s)  {question[:60]}")
        if ok:
            passed += 1
        else:
            print(f"        missing keywords: {missing}")
            print(f"        answer preview:  {answer[:200].replace(chr(10), ' ')}...")
            failed_cases.append({"id": case_id, "missing": missing, "preview": answer[:200]})

    elapsed = time.time() - started
    total = len(test_set)
    print(f"\n{passed}/{total} passed  ({elapsed:.1f}s total)")

    return 0 if not failed_cases else 1


if __name__ == "__main__":
    sys.exit(main())
