"""Tests for AgentTracer and ActionContext."""

import threading
import time

import pytest

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.validators import is_valid_hash
from agentlens.sdk.tracer import AgentTracer


def _make_tracer() -> AgentTracer:
    return AgentTracer(
        agent_type="test_agent",
        task_category=TaskCategory.OTHER,
        model_used="test-model",
    )


class TestSessionLifecycle:
    def test_basic_lifecycle(self):
        tracer = _make_tracer()
        session_id = tracer.start_session("Test task description")
        assert session_id is not None

        tracer.record_action(
            action_type=ActionType.READ,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input="read file contents",
            output_summary="file read ok",
            duration_ms=10,
        )
        tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input="analyze data",
            output_summary="analysis complete",
            duration_ms=50,
        )
        tracer.record_action(
            action_type=ActionType.WRITE,
            autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
            outcome=ActionOutcome.SUCCESS,
            raw_input="write results",
            output_summary="results written",
            duration_ms=20,
        )

        trace = tracer.end_session(SessionOutcome.SUCCESS)
        assert trace.session_id == session_id
        assert trace.agent_id == "test_agent"
        assert trace.task_category == TaskCategory.OTHER
        assert len(trace.actions) == 3
        assert trace.session_outcome == SessionOutcome.SUCCESS
        assert trace.end_time >= trace.start_time
        assert trace.metadata["model_used"] == "test-model"
        assert "task_description_hash" in trace.metadata

    def test_end_session_no_actions_raises(self):
        tracer = _make_tracer()
        tracer.start_session("empty session")
        with pytest.raises(ValueError, match="no actions"):
            tracer.end_session(SessionOutcome.SUCCESS)


class TestActionContext:
    def test_context_manager_latency(self):
        tracer = _make_tracer()
        tracer.start_session("latency test")

        with tracer.action(
            action_type=ActionType.EXECUTE,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input="some input",
        ) as ctx:
            time.sleep(0.01)
            ctx.set_output_summary("done")

        trace = tracer.end_session(SessionOutcome.SUCCESS)
        assert trace.actions[0].duration_ms > 0

    def test_error_capture(self):
        tracer = _make_tracer()
        tracer.start_session("error test")

        with pytest.raises(ValueError, match="test error"):
            with tracer.action(
                action_type=ActionType.EXECUTE,
                autonomy_level=AutonomyLevel.FULL_AUTO,
                raw_input="will fail",
            ):
                raise ValueError("test error")

        trace = tracer.end_session(SessionOutcome.FAILURE)
        action = trace.actions[0]
        assert action.outcome == ActionOutcome.FAILURE
        assert action.metadata["error_type"] == "ValueError"

    def test_set_outcome_override(self):
        tracer = _make_tracer()
        tracer.start_session("outcome test")

        with tracer.action(
            action_type=ActionType.EXECUTE,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input="partial work",
        ) as ctx:
            ctx.set_outcome(ActionOutcome.PARTIAL)
            ctx.set_output_summary("only half done")

        trace = tracer.end_session(SessionOutcome.PARTIAL)
        assert trace.actions[0].outcome == ActionOutcome.PARTIAL


class TestInputHashing:
    def test_raw_input_is_hashed(self):
        tracer = _make_tracer()
        tracer.start_session("hash test")

        raw = "This is sensitive user input that should be hashed"
        tracer.record_action(
            action_type=ActionType.READ,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input=raw,
            output_summary="ok",
            duration_ms=5,
        )

        trace = tracer.end_session(SessionOutcome.SUCCESS)
        action = trace.actions[0]
        assert is_valid_hash(action.input_hash)
        assert raw not in action.input_hash
        assert raw not in trace.to_json()


class TestEscalation:
    def test_escalation_recording(self):
        tracer = _make_tracer()
        tracer.start_session("escalation test")

        tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input="first action",
            output_summary="done",
            duration_ms=10,
        )

        tracer.record_escalation(
            reason=EscalationReason.CONFIDENCE_LOW,
            context_summary="Not sure about the answer",
        )
        tracer.record_escalation(
            reason=EscalationReason.POLICY_REQUIRED,
            context_summary="Need approval for destructive action",
        )

        trace = tracer.end_session(SessionOutcome.ESCALATED)
        assert len(trace.escalations) == 2
        assert trace.escalations[0].reason == EscalationReason.CONFIDENCE_LOW
        assert trace.escalations[1].reason == EscalationReason.POLICY_REQUIRED

    def test_escalation_no_actions_no_id_raises(self):
        tracer = _make_tracer()
        tracer.start_session("no action escalation")
        with pytest.raises(ValueError, match="No actions recorded"):
            tracer.record_escalation(
                reason=EscalationReason.CONFIDENCE_LOW,
                context_summary="test",
            )


class TestThreadSafety:
    def test_concurrent_recording(self):
        tracer = _make_tracer()
        tracer.start_session("thread safety test")

        errors: list[Exception] = []

        def record_actions(thread_id: int) -> None:
            try:
                for i in range(10):
                    tracer.record_action(
                        action_type=ActionType.EXECUTE,
                        autonomy_level=AutonomyLevel.FULL_AUTO,
                        outcome=ActionOutcome.SUCCESS,
                        raw_input=f"thread-{thread_id}-action-{i}",
                        output_summary=f"done-{thread_id}-{i}",
                        duration_ms=1,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_actions, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        trace = tracer.end_session(SessionOutcome.SUCCESS)
        assert len(trace.actions) == 30


class TestValidation:
    def test_trace_passes_pydantic_validation(self):
        tracer = _make_tracer()
        tracer.start_session("validation test")

        tracer.record_action(
            action_type=ActionType.READ,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input="test input",
            output_summary="test output",
            duration_ms=5,
        )

        trace = tracer.end_session(SessionOutcome.SUCCESS)
        # Round-trip through JSON to ensure full validation
        json_str = trace.to_json()
        from agentlens.schema.trace import SessionTrace

        reloaded = SessionTrace.from_json(json_str)
        assert reloaded.session_id == trace.session_id
        assert len(reloaded.actions) == 1
