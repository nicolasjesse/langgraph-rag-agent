"""Mocked unit tests for agent.py.

These tests verify graph wiring + node logic with both Chroma and the LLMs
mocked. They do NOT call Anthropic or OpenAI — runnable in CI without keys.

Real-API + answer-quality coverage belongs in the Day-3 test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

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

    def __init__(self, schema, parent_llm):
        self._schema = schema
        self._parent = parent_llm

    def invoke(self, messages):
        name = self._schema.__name__
        if name == "Classification":
            return self._schema(
                category=self._parent.category,
                reason="mocked classification",
            )
        if name == "Verification":
            passes = self._parent.next_verifier_pass()
            return self._schema(
                passes=passes,
                reason="looks good" if passes else "missing detail X",
            )
        raise ValueError(f"Unknown schema: {name}")


class FakeLLM:
    """Stands in for ChatAnthropic. Supports both .invoke() and .with_structured_output()."""

    def __init__(self, *, category: str = "needs_retrieval", verifier_passes=True):
        self.category = category
        self._verifier_seq = (
            [verifier_passes] * 100 if isinstance(verifier_passes, bool) else list(verifier_passes)
        )
        self._verifier_idx = 0
        # Inspection counters
        self.planner_calls = 0
        self.direct_answer_calls = 0
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        last_user = messages[-1][1] if messages else ""
        if "Context:" in last_user:
            self.planner_calls += 1
        else:
            self.direct_answer_calls += 1
        return FakeResponse("Mocked answer about LangGraph [mock_a.md].")

    def with_structured_output(self, schema):
        return FakeStructuredLLM(schema, self)

    def next_verifier_pass(self) -> bool:
        if self._verifier_idx < len(self._verifier_seq):
            v = self._verifier_seq[self._verifier_idx]
            self._verifier_idx += 1
            return v
        return True


# --- Tests -------------------------------------------------------------------


def test_supervisor_classifies_query():
    """supervisor_node should write the category to state."""
    fake = FakeLLM(category="needs_retrieval")
    with patch.object(agent, "ChatAnthropic", return_value=fake):
        result = agent.supervisor_node({"query": "What is LangGraph?"})

    assert result == {"category": "needs_retrieval"}


def test_retrieval_node_shapes_chroma_results():
    with patch.object(agent, "_get_collection", return_value=FakeCollection()):
        result = agent.retrieval_node({"query": "anything"})

    docs = result["retrieved_docs"]
    assert len(docs) == 2
    assert docs[0] == {"text": "First mock chunk.", "source": "mock_a.md", "chunk_index": 0}


def test_planner_node_includes_feedback_on_retry():
    """planner_node should reference the verifier's feedback when iteration_count > 0."""
    fake = FakeLLM()
    state = {
        "query": "How does retrieval work?",
        "category": "needs_retrieval",
        "retrieved_docs": [
            {"text": "Some doc text.", "source": "mock_a.md", "chunk_index": 0},
        ],
        "answer": "",
        "iteration_count": 1,
        "verifier_passes": False,
        "verifier_feedback": "answer made up an API method",
    }
    with patch.object(agent, "ChatAnthropic", return_value=fake):
        agent.planner_node(state)

    user_message = fake.last_messages[1][1]
    assert "How does retrieval work?" in user_message
    assert "answer made up an API method" in user_message
    assert "previous attempt was rejected" in user_message


def test_verifier_node_passes_writes_state():
    fake = FakeLLM(verifier_passes=True)
    state = {
        "query": "q",
        "retrieved_docs": [{"text": "doc", "source": "a.md", "chunk_index": 0}],
        "answer": "an answer",
    }
    with patch.object(agent, "ChatAnthropic", return_value=fake):
        result = agent.verifier_node(state)

    assert result["verifier_passes"] is True
    assert result["verifier_feedback"] == "looks good"
    assert result["iteration_count"] == 1


def test_verifier_node_fails_increments_iteration_count():
    fake = FakeLLM(verifier_passes=False)
    state = {
        "query": "q",
        "retrieved_docs": [{"text": "doc", "source": "a.md", "chunk_index": 0}],
        "answer": "bad answer",
        "iteration_count": 1,
    }
    with patch.object(agent, "ChatAnthropic", return_value=fake):
        result = agent.verifier_node(state)

    assert result["verifier_passes"] is False
    assert result["iteration_count"] == 2


def test_full_graph_routes_simple_query_skips_retrieval():
    mock_collection = MagicMock(wraps=FakeCollection())
    fake = FakeLLM(category="simple")
    with patch.object(agent, "_get_collection", return_value=mock_collection), \
         patch.object(agent, "ChatAnthropic", return_value=fake):
        graph = agent.build_graph()
        result = graph.invoke({"query": "hi there"})

    assert result["category"] == "simple"
    assert result["answer"]
    mock_collection.query.assert_not_called()


def test_full_graph_passes_first_attempt_no_loop():
    """needs_retrieval + verifier passes once → planner runs exactly once."""
    fake = FakeLLM(category="needs_retrieval", verifier_passes=True)
    with patch.object(agent, "_get_collection", return_value=FakeCollection()), \
         patch.object(agent, "ChatAnthropic", return_value=fake):
        graph = agent.build_graph()
        result = graph.invoke({"query": "What is LangGraph?"})

    assert result["verifier_passes"] is True
    assert result["iteration_count"] == 1
    assert fake.planner_calls == 1


def test_full_graph_loops_then_passes_on_retry():
    """First verifier check fails, second passes → planner runs twice."""
    fake = FakeLLM(category="needs_retrieval", verifier_passes=[False, True])
    with patch.object(agent, "_get_collection", return_value=FakeCollection()), \
         patch.object(agent, "ChatAnthropic", return_value=fake):
        graph = agent.build_graph()
        result = graph.invoke({"query": "What is LangGraph?"})

    assert result["verifier_passes"] is True
    assert result["iteration_count"] == 2
    assert fake.planner_calls == 2


def test_full_graph_terminates_at_max_iterations():
    """Verifier always fails → graph stops after MAX_ITERATIONS planner attempts."""
    fake = FakeLLM(category="needs_retrieval", verifier_passes=False)
    with patch.object(agent, "_get_collection", return_value=FakeCollection()), \
         patch.object(agent, "ChatAnthropic", return_value=fake):
        graph = agent.build_graph()
        result = graph.invoke({"query": "What is LangGraph?"})

    assert result["verifier_passes"] is False
    assert result["iteration_count"] == agent.MAX_ITERATIONS
    assert fake.planner_calls == agent.MAX_ITERATIONS


def test_agent_state_has_expected_fields():
    assert set(agent.AgentState.__annotations__.keys()) == {
        "query",
        "category",
        "retrieved_docs",
        "answer",
        "iteration_count",
        "verifier_passes",
        "verifier_feedback",
    }


def test_checkpointer_persists_state_under_thread_id():
    """With a checkpointer, get_state(config) should return the final state for a thread."""
    fake = FakeLLM(category="needs_retrieval", verifier_passes=True)
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "test-thread"}}

    with patch.object(agent, "_get_collection", return_value=FakeCollection()), \
         patch.object(agent, "ChatAnthropic", return_value=fake):
        graph = agent.build_graph(checkpointer=checkpointer)
        graph.invoke({"query": "What is LangGraph?"}, config=config)

    snapshot = graph.get_state(config)
    assert snapshot.values["query"] == "What is LangGraph?"
    assert snapshot.values["verifier_passes"] is True
    assert snapshot.values["iteration_count"] == 1
