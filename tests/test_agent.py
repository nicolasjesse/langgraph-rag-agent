"""Mocked unit tests for agent.py.

These tests verify graph wiring + node logic with both Chroma and the LLMs
mocked. They do NOT call Anthropic or OpenAI — runnable in CI without keys.

Real-API + answer-quality coverage belongs in the Day-3 evaluation harness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import agent


# --- Fakes -------------------------------------------------------------------


class FakeCollection:
    """Stands in for a Chroma collection."""

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


class FakeStructuredLLM:
    """Returned by FakeLLM.with_structured_output(); produces a schema instance."""

    def __init__(self, schema, category: str):
        self._schema = schema
        self._category = category

    def invoke(self, messages):
        return self._schema(category=self._category, reason="mocked classification")


class FakeLLM:
    """Stands in for ChatAnthropic. Supports both chat (.invoke) and structured-output paths."""

    def __init__(self, *args, category: str = "needs_retrieval", **kwargs):
        self._category = category
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse("Mocked answer about LangGraph [mock_a.md].")

    def with_structured_output(self, schema):
        return FakeStructuredLLM(schema, self._category)


def _make_chat_anthropic_factory(category: str = "needs_retrieval"):
    """Return a fake ChatAnthropic constructor that yields FakeLLM with the given category."""

    def factory(*args, **kwargs):
        return FakeLLM(category=category)

    return factory


# --- Tests -------------------------------------------------------------------


def test_supervisor_classifies_query():
    """supervisor_node should write the category to state."""
    with patch.object(agent, "ChatAnthropic", side_effect=_make_chat_anthropic_factory("needs_retrieval")):
        result = agent.supervisor_node({"query": "What is LangGraph?"})

    assert result == {"category": "needs_retrieval"}


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
        "category": "needs_retrieval",
        "retrieved_docs": [
            {"text": "Some doc text.", "source": "mock_a.md", "chunk_index": 0},
        ],
        "answer": "",
    }
    with patch.object(agent, "ChatAnthropic", return_value=fake_llm):
        result = agent.planner_node(state)

    assert result == {"answer": "Mocked answer about LangGraph [mock_a.md]."}

    user_message = fake_llm.last_messages[1][1]
    assert "Some doc text." in user_message
    assert "How does retrieval work?" in user_message
    assert "[mock_a.md]" in user_message


def test_direct_answer_node_uses_query_only():
    """direct_answer_node should not need retrieval; just ask the LLM."""
    fake_llm = FakeLLM()
    with patch.object(agent, "ChatAnthropic", return_value=fake_llm):
        result = agent.direct_answer_node({"query": "Hi!", "category": "simple"})

    assert "answer" in result
    assert fake_llm.last_messages[1] == ("user", "Hi!")


def test_full_graph_routes_needs_retrieval_through_planner():
    """When supervisor returns 'needs_retrieval', graph should run retrieval + planner."""
    mock_collection = MagicMock(wraps=FakeCollection())
    with patch.object(agent, "_get_collection", return_value=mock_collection), \
         patch.object(agent, "ChatAnthropic", side_effect=_make_chat_anthropic_factory("needs_retrieval")):
        graph = agent.build_graph()
        result = graph.invoke({"query": "What is LangGraph?"})

    assert result["category"] == "needs_retrieval"
    assert len(result["retrieved_docs"]) == 2
    assert result["answer"] == "Mocked answer about LangGraph [mock_a.md]."
    mock_collection.query.assert_called_once()


def test_full_graph_routes_simple_query_skips_retrieval():
    """When supervisor returns 'simple', graph should skip retrieval entirely."""
    mock_collection = MagicMock(wraps=FakeCollection())
    with patch.object(agent, "_get_collection", return_value=mock_collection), \
         patch.object(agent, "ChatAnthropic", side_effect=_make_chat_anthropic_factory("simple")):
        graph = agent.build_graph()
        result = graph.invoke({"query": "hi there"})

    assert result["category"] == "simple"
    assert result["answer"]
    mock_collection.query.assert_not_called()
    assert result.get("retrieved_docs") in (None, [])


def test_agent_state_has_expected_fields():
    """Schema sanity-check — guards against accidental field rename."""
    assert set(agent.AgentState.__annotations__.keys()) == {
        "query",
        "category",
        "retrieved_docs",
        "answer",
    }
