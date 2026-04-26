"""AWS Lambda entry point for the RAG agent.

Receives an API Gateway proxy event with a JSON body of the form:
    {"query": "...", "thread_id": "optional-conversation-id"}

Returns a JSON response with the agent's answer plus metadata.

Cold-start work (graph build, chroma_db copy to /tmp) happens at module
import so subsequent invocations on a warm container reuse the graph.
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path

# --- Cold-start setup --------------------------------------------------------

# /var/task is the read-only directory where the image's code + baked corpus
# live. ChromaDB needs write access (lockfiles, WAL), so on first run we copy
# the pre-built collection to /tmp (the only writable location in Lambda).
_BAKED_CHROMA = Path("/var/task/chroma_db")
_RUNTIME_CHROMA = Path("/tmp/chroma_db")

if _BAKED_CHROMA.exists() and not _RUNTIME_CHROMA.exists():
    shutil.copytree(_BAKED_CHROMA, _RUNTIME_CHROMA)

os.environ.setdefault("CHROMA_DB_PATH", str(_RUNTIME_CHROMA))
os.environ.setdefault("HITL_ENABLED", "false")  # stateless runtime

from agent import build_graph  # noqa: E402  (import after env setup)

# Build the graph once per cold-start container — reused across warm invokes.
_GRAPH = build_graph(checkpointer=None)


# --- Helpers -----------------------------------------------------------------


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


# --- Handler -----------------------------------------------------------------


def handler(event, context):
    # API Gateway proxy events have body as a JSON string.
    raw_body = event.get("body") or "{}"
    if isinstance(raw_body, str):
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return _response(400, {"error": "Body is not valid JSON."})
    else:
        body = raw_body

    query = (body.get("query") or "").strip()
    if not query:
        return _response(400, {"error": "Field 'query' is required."})

    try:
        result = _GRAPH.invoke({"query": query})
    except Exception as e:
        # Log full trace for CloudWatch; return a safe message.
        print(f"agent invoke failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return _response(500, {"error": "Agent failed — see CloudWatch logs."})

    return _response(200, {
        "answer": result.get("answer"),
        "category": result.get("category"),
        "iteration_count": result.get("iteration_count"),
        "verifier_passes": result.get("verifier_passes"),
        "verifier_feedback": result.get("verifier_feedback"),
        "sources": sorted({d["source"] for d in result.get("retrieved_docs") or []}),
    })
