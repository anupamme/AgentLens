"""Failure injection — wraps AgentTracer to simulate failure scenarios."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

from agentlens.schema.enums import ActionOutcome, ActionType, AutonomyLevel
from agentlens.schema.trace import ActionRecord
from agentlens.sdk.action_context import ActionContext
from agentlens.sdk.tracer import AgentTracer
from agentlens.workloads.generator import FailureMode


class FailableActionContext:
    """Wraps ActionContext to inject failures on exit based on failure mode."""

    def __init__(
        self,
        inner: ActionContext,
        failure_mode: FailureMode,
    ) -> None:
        self._inner = inner
        self._failure_mode = failure_mode

    def set_output_summary(self, summary: str) -> None:
        self._inner.set_output_summary(summary)

    def set_outcome(self, outcome: ActionOutcome) -> None:
        self._inner.set_outcome(outcome)

    def __enter__(self) -> FailableActionContext:
        self._inner.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._apply_failure()
        self._inner.__exit__(exc_type, exc_val, exc_tb)

    def _apply_failure(self) -> None:
        if self._failure_mode == FailureMode.TOOL_TIMEOUT:
            delay = random.uniform(2.0, 10.0)
            time.sleep(delay)
            self._inner.set_outcome(ActionOutcome.TIMEOUT)
            self._inner.set_output_summary(f"timeout after {delay:.1f}s")

        elif self._failure_mode == FailureMode.AMBIGUOUS_INPUT:
            self._inner.set_outcome(ActionOutcome.PARTIAL)
            self._inner.set_output_summary(
                "partial result due to ambiguous input interpretation"
            )

        elif self._failure_mode == FailureMode.CONFLICTING_CONSTRAINTS:
            self._inner.set_outcome(ActionOutcome.FAILURE)
            self._inner.set_output_summary(
                "failed due to contradictory requirements in constraints"
            )

        elif self._failure_mode == FailureMode.SAFETY_BOUNDARY:
            self._inner.set_outcome(ActionOutcome.SKIPPED)
            self._inner.set_output_summary(
                "action skipped: safety boundary reached"
            )

        elif self._failure_mode == FailureMode.PARTIAL_FAILURE:
            if random.random() < 0.5:
                self._inner.set_outcome(ActionOutcome.FAILURE)
                self._inner.set_output_summary("intermittent failure")
            # else: keep the default SUCCESS outcome


class InstrumentedTracer:
    """Wraps AgentTracer via composition, injecting failures into action() calls."""

    def __init__(self, tracer: AgentTracer, failure_mode: FailureMode) -> None:
        self._tracer = tracer
        self._failure_mode = failure_mode

    def start_session(self, task_description: str) -> str:
        return self._tracer.start_session(task_description)

    def action(
        self,
        action_type: ActionType,
        autonomy_level: AutonomyLevel,
        raw_input: str,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FailableActionContext:
        inner = self._tracer.action(
            action_type=action_type,
            autonomy_level=autonomy_level,
            raw_input=raw_input,
            tool_name=tool_name,
            metadata=metadata,
        )
        return FailableActionContext(inner, self._failure_mode)

    def record_escalation(self, reason: Any, context_summary: str, triggering_action_id: str | None = None) -> Any:
        return self._tracer.record_escalation(reason, context_summary, triggering_action_id)

    def end_session(self, outcome: Any, user_satisfaction_proxy: float | None = None) -> Any:
        return self._tracer.end_session(outcome, user_satisfaction_proxy)

    @property
    def tracer(self) -> AgentTracer:
        return self._tracer


class FailureInjector:
    """Factory for wrapping tools and tracers with failure injection."""

    @staticmethod
    def wrap_tool(tool_fn: Callable[..., Any], failure_mode: FailureMode) -> Callable[..., Any]:
        """Wrap a tool function to simulate failures."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if failure_mode == FailureMode.TOOL_TIMEOUT:
                delay = random.uniform(2.0, 10.0)
                time.sleep(delay)
                raise TimeoutError(f"Tool timed out after {delay:.1f}s")

            elif failure_mode == FailureMode.AMBIGUOUS_INPUT:
                kwargs["_injected_context"] = (
                    "WARNING: multiple interpretations possible. "
                    "Conflicting context detected in input data."
                )
                return tool_fn(*args, **kwargs)

            elif failure_mode == FailureMode.CONFLICTING_CONSTRAINTS:
                result = tool_fn(*args, **kwargs)
                return {"original": result, "contradictory": "opposite_result", "_conflict": True}

            elif failure_mode == FailureMode.PARTIAL_FAILURE:
                if random.random() < 0.5:
                    raise RuntimeError("Intermittent tool failure")
                return tool_fn(*args, **kwargs)

            elif failure_mode == FailureMode.SAFETY_BOUNDARY:
                raise PermissionError("Action blocked by safety boundary")

            return tool_fn(*args, **kwargs)

        return wrapped

    @staticmethod
    def wrap_tracer(tracer: AgentTracer, failure_mode: FailureMode) -> InstrumentedTracer:
        """Wrap an AgentTracer to inject failures into action contexts."""
        return InstrumentedTracer(tracer, failure_mode)
