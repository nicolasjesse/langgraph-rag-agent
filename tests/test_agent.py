"""Mocked unit tests for agent.py.

These tests verify that the graph wiring and node logic are correct.
They do NOT call Anthropic or OpenAI — both are mocked, so the tests
are fast, free, and runnable in CI without API keys.

Real API + answer-quality verification belongs in evals/ (Day 3).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import agent


# --- Fakes -------------------------------------------------------------------


class FakeCollection:
    """Stands in for a Chroma collection. Returns canned results."""

    def query(self, query_texts, n_results):
        return {
            "documents": [["First mock chunk.", "Second mock chunk."]],
            "metadatas": [
                [
                    {"source": "mock_a.md", "chunk_index": 0},
                    {"source": "mock_b.md", "chunk_index": 7},
                ]
            ],
            "distances": [[0.10, 0.25]],
        }


class FakeResponse:
    """Mimics LangChain's AIMessage shape — just needs `.content`."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stands in for ChatAnthropic. Records the prompt it was called with."""

    def __init__(self, *args, **kwargs):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse("Mocked answer about LangGraph [mock_a.md].")


# --- Tests -------------------------------------------------------------------


def test_retrieval_node_shapes_chroma_results():
    """retrieval_node should transform Chroma's response into a list of dicts."""
    with patch.object(agent, "_get_collection", return_value=FakeCollection()):
        result = agent.retrieval_node({"query": "anything"})

    assert "retrieved_docs" in result
    docs = result["retrieved_docs"]
    assert len(docs) == 2
    assert docs[0] == {"text": "First mock chunk.", "source": "mock_a.md", "chunk_index": 0}
    assert docs[1] == {"text": "Second mock chunk.", "source": "mock_b.md", "chunk_index": 7}


def test_planner_node_passes_context_and_question_to_llm():
    """planner_node should build a prompt that includes both the docs and the query."""
    fake_llm = FakeLLM()
    state = {
        "query": "How does retrieval work?",
        "retrieved_docs": [
            {"text": "Some doc text.", "source": "mock_a.md", "chunk_index": 0},
        ],
        "answer": "",
    }
    with patch.object(agent, "ChatAnthropic", return_value=fake_llm):
        result = agent.planner_node(state)

    assert result == {"answer": "Mocked answer about LangGraph [mock_a.md]."}

    # Confirm the prompt actually contained the doc and the question.
    user_message = fake_llm.last_messages[1][1]   # ("user", "...")
    assert "Some doc text." in user_message
    assert "How does retrieval work?" in user_message
    assert "[mock_a.md]" in user_message  # citation label is present


def test_full_graph_runs_end_to_end_with_mocks():
    """Compile the graph and invoke it; both Chroma and the LLM are mocked."""
    with patch.object(agent, "_get_collection", return_value=FakeCollection()), \
         patch.object(agent, "ChatAnthropic", return_value=FakeLLM()):
        graph = agent.build_graph()
        result = graph.invoke({"query": "What is a graph?"})

    # Final state should have all three fields populated.
    assert result["query"] == "What is a graph?"
    assert len(result["retrieved_docs"]) == 2
    assert result["answer"] == "Mocked answer about LangGraph [mock_a.md]."


def test_agent_state_has_expected_fields():
    """Cheap schema sanity-check — guards against accidental field rename."""
    assert set(agent.AgentState.__annotations__.keys()) == {
        "query",
        "retrieved_docs",
        "answer",
    }
