"""LangChain callback handler for automatic AgentLens tracing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "langchain-core is required for the LangChain integration. "
        "Install it with: pip install agentlens[langchain]"
    )

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import SessionTrace
from agentlens.sdk.tracer import AgentTracer


class AgentLensCallbackHandler(BaseCallbackHandler):
    """Drop-in LangChain callback handler that produces AgentLens traces."""

    def __init__(
        self,
        agent_type: str = "langchain_agent",
        task_category: TaskCategory = TaskCategory.OTHER,
        model_used: str | None = None,
        task_description: str = "LangChain agent session",
    ) -> None:
        super().__init__()
        self._tracer = AgentTracer(
            agent_type=agent_type,
            task_category=task_category,
            model_used=model_used,
        )
        self._task_description = task_description
        self._chain_depth = 0
        self._session_started = False
        self._session_ended = False
        self._current_llm_run: dict[str, int] = {}  # run_id -> start_time_ns
        self._current_tool_run: dict[str, int] = {}
        self._had_error = False

    def _ensure_session(self) -> None:
        if not self._session_started:
            self._tracer.start_session(self._task_description)
            self._session_started = True

    # -- Chain callbacks (session lifecycle) --

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self._chain_depth += 1
        if self._chain_depth == 1:
            self._ensure_session()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self._chain_depth = max(0, self._chain_depth - 1)
        if self._chain_depth == 0 and not self._session_ended:
            self._session_ended = True
            outcome = SessionOutcome.FAILURE if self._had_error else SessionOutcome.SUCCESS
            self._tracer.end_session(outcome)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self._had_error = True
        self._chain_depth = max(0, self._chain_depth - 1)
        if self._chain_depth == 0 and not self._session_ended:
            self._session_ended = True
            self._tracer.end_session(SessionOutcome.FAILURE)

    # -- LLM callbacks --

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        import time

        self._ensure_session()
        run_id = str(kwargs.get("run_id", uuid4()))
        self._current_llm_run[run_id] = time.monotonic_ns()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        import time

        run_id = str(kwargs.get("run_id", ""))
        start_ns = self._current_llm_run.pop(run_id, time.monotonic_ns())
        duration_ms = max(int((time.monotonic_ns() - start_ns) // 1_000_000), 0)

        output_text = ""
        if hasattr(response, "generations") and response.generations:
            first_gen = response.generations[0]
            if first_gen and hasattr(first_gen[0], "text"):
                output_text = first_gen[0].text[:500]

        self._tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input=f"llm_call_{run_id}",
            output_summary=output_text or "llm response",
            duration_ms=duration_ms,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        import time

        self._had_error = True
        run_id = str(kwargs.get("run_id", ""))
        start_ns = self._current_llm_run.pop(run_id, time.monotonic_ns())
        duration_ms = max(int((time.monotonic_ns() - start_ns) // 1_000_000), 0)

        self._tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.FAILURE,
            raw_input=f"llm_call_{run_id}",
            output_summary=f"error: {type(error).__name__}",
            duration_ms=duration_ms,
            metadata={"error_type": type(error).__name__},
        )

    # -- Tool callbacks --

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        import time

        self._ensure_session()
        run_id = str(kwargs.get("run_id", uuid4()))
        self._current_tool_run[run_id] = time.monotonic_ns()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        import time

        run_id = str(kwargs.get("run_id", ""))
        start_ns = self._current_tool_run.pop(run_id, time.monotonic_ns())
        duration_ms = max(int((time.monotonic_ns() - start_ns) // 1_000_000), 0)

        tool_name = kwargs.get("name", "unknown_tool")

        self._tracer.record_action(
            action_type=ActionType.EXECUTE,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input=f"tool_call_{run_id}",
            output_summary=str(output)[:500],
            duration_ms=duration_ms,
            tool_name=str(tool_name),
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        import time

        self._had_error = True
        run_id = str(kwargs.get("run_id", ""))
        start_ns = self._current_tool_run.pop(run_id, time.monotonic_ns())
        duration_ms = max(int((time.monotonic_ns() - start_ns) // 1_000_000), 0)

        tool_name = kwargs.get("name", "unknown_tool")

        self._tracer.record_action(
            action_type=ActionType.EXECUTE,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.FAILURE,
            raw_input=f"tool_call_{run_id}",
            output_summary=f"error: {type(error).__name__}",
            duration_ms=duration_ms,
            tool_name=str(tool_name),
            metadata={"error_type": type(error).__name__},
        )

    def get_trace(self) -> SessionTrace | None:
        return self._tracer._trace
