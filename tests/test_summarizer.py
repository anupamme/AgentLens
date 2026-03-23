"""Tests for the per-session summarizer."""

from datetime import datetime, timezone

import pytest

from agentlens.aggregation.models import SessionSummary
from agentlens.aggregation.summarizer import MockSummarizer
from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, EscalationEvent, SessionTrace
from agentlens.utils.hashing import hash_input


def make_action(**overrides):
    defaults = {
        "action_id": "act-001",
        "action_type": ActionType.READ,
        "autonomy_level": AutonomyLevel.FULL_AUTO,
        "outcome": ActionOutcome.SUCCESS,
        "timestamp": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "duration_ms": 100,
        "input_hash": hash_input("test input"),
        "output_summary": "Test output",
    }
    defaults.update(overrides)
    return ActionRecord(**defaults)


def make_session(**overrides):
    defaults = {
        "session_id": "sess-001",
        "agent_id": "test-agent",
        "task_category": TaskCategory.CODE_REVIEW,
        "session_outcome": SessionOutcome.SUCCESS,
        "start_time": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        "actions": [make_action()],
    }
    defaults.update(overrides)
    return SessionTrace(**defaults)


class TestMockSummarizer:
    @pytest.mark.asyncio
    async def test_valid_summaries_for_different_traces(self):
        """Feed 3 different traces, assert all summaries validate against SessionSummary."""
        summarizer = MockSummarizer()

        traces = [
            make_session(
                session_id="s1",
                agent_id="code-reviewer",
                task_category=TaskCategory.CODE_REVIEW,
                actions=[
                    make_action(action_id="a1", action_type=ActionType.READ),
                    make_action(action_id="a2", action_type=ActionType.REASON),
                    make_action(action_id="a3", action_type=ActionType.WRITE),
                ],
            ),
            make_session(
                session_id="s2",
                agent_id="researcher",
                task_category=TaskCategory.RESEARCH,
                session_outcome=SessionOutcome.PARTIAL,
                actions=[
                    make_action(action_id="a1", action_type=ActionType.SEARCH),
                    make_action(action_id="a2", action_type=ActionType.REASON),
                ],
            ),
            make_session(
                session_id="s3",
                agent_id="sys-admin",
                task_category=TaskCategory.SYSTEM_ADMIN,
                actions=[
                    make_action(
                        action_id="a1",
                        action_type=ActionType.EXECUTE,
                        tool_name="bash",
                    ),
                ],
            ),
        ]

        for trace in traces:
            summary = await summarizer.summarize(trace)
            assert isinstance(summary, SessionSummary)
            assert summary.session_id == trace.session_id
            assert summary.agent_type == trace.agent_id
            assert summary.task_category == trace.task_category

    @pytest.mark.asyncio
    async def test_autonomy_distribution_sums_to_one(self):
        """Assert autonomy_distribution values sum to ~1.0."""
        summarizer = MockSummarizer()

        actions = [
            make_action(action_id="a1", autonomy_level=AutonomyLevel.FULL_AUTO),
            make_action(action_id="a2", autonomy_level=AutonomyLevel.FULL_AUTO),
            make_action(action_id="a3", autonomy_level=AutonomyLevel.HUMAN_CONFIRMED),
            make_action(action_id="a4", autonomy_level=AutonomyLevel.HUMAN_DRIVEN),
            make_action(action_id="a5", autonomy_level=AutonomyLevel.AUTO_WITH_AUDIT),
        ]
        trace = make_session(actions=actions)
        summary = await summarizer.summarize(trace)

        total = sum(summary.autonomy_distribution.values())
        assert abs(total - 1.0) < 0.001

        # Verify individual fractions
        assert abs(summary.autonomy_distribution["fully_autonomous"] - 0.4) < 0.001
        assert abs(summary.autonomy_distribution["auto_with_audit"] - 0.2) < 0.001
        assert abs(summary.autonomy_distribution["human_confirmed"] - 0.2) < 0.001
        assert abs(summary.autonomy_distribution["human_driven"] - 0.2) < 0.001

    @pytest.mark.asyncio
    async def test_oversight_gap_score_computation(self):
        """Create a trace with known consequential/autonomous actions, verify score."""
        summarizer = MockSummarizer()

        actions = [
            # Consequential + fully autonomous → counts toward gap
            make_action(
                action_id="a1",
                action_type=ActionType.WRITE,
                autonomy_level=AutonomyLevel.FULL_AUTO,
            ),
            # Consequential + fully autonomous → counts toward gap
            make_action(
                action_id="a2",
                action_type=ActionType.EXECUTE,
                autonomy_level=AutonomyLevel.FULL_AUTO,
            ),
            # Consequential + human confirmed → does NOT count toward gap
            make_action(
                action_id="a3",
                action_type=ActionType.COMMUNICATE,
                autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
            ),
            # Non-consequential → ignored
            make_action(
                action_id="a4",
                action_type=ActionType.READ,
                autonomy_level=AutonomyLevel.FULL_AUTO,
            ),
        ]
        trace = make_session(actions=actions)
        summary = await summarizer.summarize(trace)

        # 3 consequential actions, 2 unsupervised → gap = 2/3
        assert summary.consequential_action_count == 3
        assert summary.unsupervised_consequential_count == 2
        assert abs(summary.oversight_gap_score - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_no_pii_leakage_in_mock(self):
        """Assert task_abstract doesn't contain any strings from raw inputs."""
        summarizer = MockSummarizer()

        # The raw input contains sensitive data, but it's hashed in the trace
        secret_data = "user password is hunter2 and email is john@example.com"
        trace = make_session(
            actions=[
                make_action(
                    action_id="a1",
                    input_hash=hash_input(secret_data),
                    output_summary="Processed request",
                ),
            ],
            metadata={"task_description_hash": hash_input("secret task")},
        )
        summary = await summarizer.summarize(trace)

        # task_abstract should not contain any raw input data
        assert "hunter2" not in summary.task_abstract
        assert "john@example.com" not in summary.task_abstract
        assert "password" not in summary.task_abstract
        assert "secret" not in summary.task_abstract

    @pytest.mark.asyncio
    async def test_batch_summarization(self):
        """Summarize 10 traces, assert 10 summaries returned."""
        summarizer = MockSummarizer()

        traces = [
            make_session(
                session_id=f"sess-{i:03d}",
                actions=[make_action(action_id=f"act-{i}")],
            )
            for i in range(10)
        ]
        summaries = await summarizer.summarize_batch(traces)

        assert len(summaries) == 10
        session_ids = {s.session_id for s in summaries}
        assert len(session_ids) == 10

    @pytest.mark.asyncio
    async def test_tool_usage_tracking(self):
        """Verify tool usage fields are correctly populated."""
        summarizer = MockSummarizer()

        actions = [
            make_action(action_id="a1", tool_name="search_tool"),
            make_action(action_id="a2", tool_name="write_tool"),
            make_action(
                action_id="a3",
                tool_name="search_tool",
                outcome=ActionOutcome.FAILURE,
                metadata={"error_type": "timeout"},
            ),
            make_action(action_id="a4"),  # no tool
        ]
        trace = make_session(actions=actions)
        summary = await summarizer.summarize(trace)

        assert sorted(summary.tools_used) == ["search_tool", "write_tool"]
        assert summary.tool_call_count == 3
        # 2 out of 3 tool calls succeeded
        assert abs(summary.tool_success_rate - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_failure_and_escalation_tracking(self):
        """Verify failure and escalation fields."""
        summarizer = MockSummarizer()

        actions = [
            make_action(action_id="a1", outcome=ActionOutcome.SUCCESS),
            make_action(
                action_id="a2",
                outcome=ActionOutcome.FAILURE,
                metadata={"error_type": "api_error"},
            ),
            make_action(
                action_id="a3",
                outcome=ActionOutcome.FAILURE,
                metadata={"error_type": "timeout"},
            ),
        ]
        escalations = [
            EscalationEvent(
                timestamp=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
                reason=EscalationReason.ERROR_REPEATED,
                action_id="a2",
                description="Repeated API errors",
            ),
            EscalationEvent(
                timestamp=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
                reason=EscalationReason.RISK_HIGH,
                action_id="a3",
                description="High risk timeout",
            ),
        ]
        trace = make_session(actions=actions, escalations=escalations)
        summary = await summarizer.summarize(trace)

        assert summary.failure_count == 2
        assert sorted(summary.failure_types) == ["api_error", "timeout"]
        assert summary.escalation_count == 2
        assert sorted(summary.escalation_reasons) == ["error_repeated", "risk_high"]
        # Both failures have escalations → graceful
        assert summary.did_fail_gracefully is True

    @pytest.mark.asyncio
    async def test_did_fail_gracefully_false(self):
        """Verify did_fail_gracefully is False when failures lack escalations."""
        summarizer = MockSummarizer()

        actions = [
            make_action(
                action_id="a1",
                outcome=ActionOutcome.FAILURE,
                metadata={"error_type": "crash"},
            ),
        ]
        # No escalation for the failure
        trace = make_session(actions=actions, session_outcome=SessionOutcome.FAILURE)
        summary = await summarizer.summarize(trace)

        assert summary.failure_count == 1
        assert summary.did_fail_gracefully is False
