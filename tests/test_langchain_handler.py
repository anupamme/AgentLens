"""Tests for the LangChain callback handler integration."""

from __future__ import annotations

from uuid import uuid4

import pytest

langchain_core = pytest.importorskip("langchain_core")

from agentlens.schema.enums import ActionOutcome, ActionType, SessionOutcome  # noqa: E402
from agentlens.schema.validators import is_valid_hash  # noqa: E402
from agentlens.sdk.integrations.langchain_handler import AgentLensCallbackHandler  # noqa: E402


class TestLangChainHandler:
    def _make_handler(self) -> AgentLensCallbackHandler:
        return AgentLensCallbackHandler(
            agent_type="test_lc_agent",
            task_description="test langchain session",
        )

    def test_session_lifecycle(self):
        handler = self._make_handler()

        handler.on_chain_start(serialized={}, inputs={"input": "hello"})

        run_id = uuid4()
        handler.on_llm_start(
            serialized={},
            prompts=["What is 2+2?"],
            run_id=run_id,
        )

        # Simulate LLM response
        class FakeGeneration:
            text = "The answer is 4"

        class FakeResponse:
            generations = [[FakeGeneration()]]

        handler.on_llm_end(FakeResponse(), run_id=run_id)

        handler.on_chain_end(outputs={"output": "done"})

        trace = handler.get_trace()
        assert trace is not None
        assert trace.session_outcome == SessionOutcome.SUCCESS
        assert len(trace.actions) == 1
        assert trace.actions[0].action_type == ActionType.REASON
        assert trace.start_time <= trace.end_time

    def test_privacy_inputs_hashed(self):
        handler = self._make_handler()
        handler.on_chain_start(serialized={}, inputs={})

        sensitive_prompt = "My SSN is 123-45-6789"
        run_id = uuid4()
        handler.on_llm_start(
            serialized={},
            prompts=[sensitive_prompt],
            run_id=run_id,
        )

        class FakeGeneration:
            text = "I cannot process SSNs"

        class FakeResponse:
            generations = [[FakeGeneration()]]

        handler.on_llm_end(FakeResponse(), run_id=run_id)
        handler.on_chain_end(outputs={})

        trace = handler.get_trace()
        assert trace is not None
        json_output = trace.to_json()
        assert sensitive_prompt not in json_output
        assert is_valid_hash(trace.actions[0].input_hash)

    def test_tool_callbacks(self):
        handler = self._make_handler()
        handler.on_chain_start(serialized={}, inputs={})

        tool_run_id = uuid4()
        handler.on_tool_start(
            serialized={},
            input_str="search query",
            run_id=tool_run_id,
        )
        handler.on_tool_end("search results here", run_id=tool_run_id, name="web_search")

        handler.on_chain_end(outputs={})

        trace = handler.get_trace()
        assert trace is not None
        assert len(trace.actions) == 1
        assert trace.actions[0].action_type == ActionType.EXECUTE

    def test_error_handling(self):
        handler = self._make_handler()
        handler.on_chain_start(serialized={}, inputs={})

        run_id = uuid4()
        handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
        handler.on_llm_error(RuntimeError("API timeout"), run_id=run_id)

        handler.on_chain_error(RuntimeError("chain failed"))

        trace = handler.get_trace()
        assert trace is not None
        assert trace.session_outcome == SessionOutcome.FAILURE
        assert trace.actions[0].outcome == ActionOutcome.FAILURE

    def test_nested_chains(self):
        handler = self._make_handler()

        # Outer chain
        handler.on_chain_start(serialized={}, inputs={})
        # Inner chain
        handler.on_chain_start(serialized={}, inputs={})

        run_id = uuid4()
        handler.on_llm_start(serialized={}, prompts=["nested"], run_id=run_id)

        class FakeGeneration:
            text = "nested response"

        class FakeResponse:
            generations = [[FakeGeneration()]]

        handler.on_llm_end(FakeResponse(), run_id=run_id)

        # Inner chain ends
        handler.on_chain_end(outputs={})
        # Session should NOT be ended yet
        assert handler.get_trace() is None

        # Outer chain ends
        handler.on_chain_end(outputs={})
        trace = handler.get_trace()
        assert trace is not None
