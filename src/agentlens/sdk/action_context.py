"""ActionContext — context manager for auto-recording action latency and errors."""

from __future__ import annotations

import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

from agentlens.schema.enums import ActionOutcome, ActionType, AutonomyLevel
from agentlens.schema.trace import ActionRecord
from agentlens.utils.hashing import hash_input
from agentlens.utils.timestamps import utc_now


class ActionContext:
    """Context manager that records action latency and errors automatically."""

    def __init__(
        self,
        action_type: ActionType,
        autonomy_level: AutonomyLevel,
        raw_input: str,
        on_complete: Callable[[ActionRecord], None],
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._action_type = action_type
        self._autonomy_level = autonomy_level
        self._raw_input = raw_input
        self._on_complete = on_complete
        self._tool_name = tool_name
        self._metadata: dict[str, Any] = metadata or {}
        self._output_summary = "completed"
        self._outcome = ActionOutcome.SUCCESS
        self._start_ns: int = 0

    def set_output_summary(self, summary: str) -> None:
        self._output_summary = summary

    def set_outcome(self, outcome: ActionOutcome) -> None:
        self._outcome = outcome

    def __enter__(self) -> ActionContext:
        self._start_ns = time.monotonic_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        elapsed_ns = time.monotonic_ns() - self._start_ns
        duration_ms = max(int(elapsed_ns // 1_000_000), 0)

        if exc_type is not None:
            self._outcome = ActionOutcome.FAILURE
            self._metadata["error_type"] = exc_type.__name__
            if self._output_summary == "completed":
                self._output_summary = f"error: {exc_type.__name__}"

        from uuid import uuid4

        record = ActionRecord(
            action_id=str(uuid4()),
            action_type=self._action_type,
            autonomy_level=self._autonomy_level,
            outcome=self._outcome,
            timestamp=utc_now(),
            duration_ms=duration_ms,
            input_hash=hash_input(self._raw_input),
            output_summary=self._output_summary,
            tool_name=self._tool_name,
            metadata=self._metadata,
        )
        self._on_complete(record)
        # Return None (falsy) — exceptions are not suppressed
