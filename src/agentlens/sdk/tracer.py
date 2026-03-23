"""AgentTracer — central session lifecycle manager."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, EscalationEvent, SessionTrace
from agentlens.sdk.action_context import ActionContext
from agentlens.sdk.writer import TraceWriter
from agentlens.utils.hashing import hash_input
from agentlens.utils.timestamps import utc_now


class AgentTracer:
    """Records a complete agent session as a validated SessionTrace."""

    def __init__(
        self,
        agent_type: str,
        task_category: TaskCategory,
        model_used: str | None = None,
    ) -> None:
        self._agent_type = agent_type
        self._task_category = task_category
        self._model_used = model_used
        self._lock = threading.Lock()
        self._actions: list[ActionRecord] = []
        self._escalations: list[EscalationEvent] = []
        self._session_id: str | None = None
        self._start_time: datetime | None = None
        self._session_metadata: dict[str, Any] = {}
        self._trace: SessionTrace | None = None

    def start_session(self, task_description: str) -> str:
        self._session_id = str(uuid4())
        self._start_time = utc_now()
        self._session_metadata["task_description_hash"] = hash_input(task_description)
        self._session_metadata["task_summary"] = task_description[:200]
        if self._model_used:
            self._session_metadata["model_used"] = self._model_used
        return self._session_id

    def record_action(
        self,
        action_type: ActionType,
        autonomy_level: AutonomyLevel,
        outcome: ActionOutcome,
        raw_input: str,
        output_summary: str,
        duration_ms: int,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionRecord:
        record = ActionRecord(
            action_id=str(uuid4()),
            action_type=action_type,
            autonomy_level=autonomy_level,
            outcome=outcome,
            timestamp=utc_now(),
            duration_ms=duration_ms,
            input_hash=hash_input(raw_input),
            output_summary=output_summary,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        with self._lock:
            self._actions.append(record)
        return record

    def action(
        self,
        action_type: ActionType,
        autonomy_level: AutonomyLevel,
        raw_input: str,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionContext:
        return ActionContext(
            action_type=action_type,
            autonomy_level=autonomy_level,
            raw_input=raw_input,
            on_complete=self._append_action,
            tool_name=tool_name,
            metadata=metadata,
        )

    def _append_action(self, record: ActionRecord) -> None:
        with self._lock:
            self._actions.append(record)

    def record_escalation(
        self,
        reason: EscalationReason,
        context_summary: str,
        triggering_action_id: str | None = None,
    ) -> EscalationEvent:
        if triggering_action_id is None:
            with self._lock:
                if not self._actions:
                    raise ValueError(
                        "No actions recorded yet; must provide triggering_action_id explicitly"
                    )
                triggering_action_id = self._actions[-1].action_id

        event = EscalationEvent(
            timestamp=utc_now(),
            reason=reason,
            action_id=triggering_action_id,
            description=context_summary[:200],
        )
        with self._lock:
            self._escalations.append(event)
        return event

    def end_session(
        self,
        outcome: SessionOutcome,
        user_satisfaction_proxy: float | None = None,
    ) -> SessionTrace:
        if not self._actions:
            raise ValueError("Cannot end session with no actions recorded")

        metadata = dict(self._session_metadata)
        if user_satisfaction_proxy is not None:
            metadata["user_satisfaction_proxy"] = user_satisfaction_proxy

        self._trace = SessionTrace(
            session_id=self._session_id or str(uuid4()),
            agent_id=self._agent_type,
            task_category=self._task_category,
            session_outcome=outcome,
            start_time=self._start_time or utc_now(),
            end_time=utc_now(),
            actions=list(self._actions),
            escalations=list(self._escalations),
            metadata=metadata,
        )
        return self._trace

    def save(self, path: str = "./traces") -> None:
        if self._trace is None:
            raise ValueError("No trace to save; call end_session() first")
        writer = TraceWriter(output_dir=path)
        writer.write_jsonl(self._trace)

    def save_json(self, path: str = "./traces") -> str:
        if self._trace is None:
            raise ValueError("No trace to save; call end_session() first")
        writer = TraceWriter(output_dir=path)
        return writer.write_json(self._trace)
