"""Tests for TraceValidator."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, SessionTrace
from agentlens.utils.hashing import hash_input
from agentlens.workloads.validator import TraceValidator


def _make_action(
    timestamp: datetime,
    outcome: ActionOutcome = ActionOutcome.SUCCESS,
    autonomy_level: AutonomyLevel = AutonomyLevel.FULL_AUTO,
    action_type: ActionType = ActionType.READ,
) -> ActionRecord:
    return ActionRecord(
        action_id=str(uuid4()),
        action_type=action_type,
        autonomy_level=autonomy_level,
        outcome=outcome,
        timestamp=timestamp,
        duration_ms=100,
        input_hash=hash_input("test input"),
        output_summary="test output",
    )


def _make_trace(
    num_actions: int = 3,
    task_category: TaskCategory = TaskCategory.CODE_REVIEW,
    session_outcome: SessionOutcome = SessionOutcome.SUCCESS,
    start_offset_minutes: int = 0,
    reverse_timestamps: bool = False,
) -> SessionTrace:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=start_offset_minutes
    )
    actions = []
    for i in range(num_actions):
        ts = base + timedelta(seconds=i * 10)
        actions.append(_make_action(timestamp=ts))

    if reverse_timestamps:
        # Swap first and last timestamps to create non-monotonic order
        actions[0], actions[-1] = (
            ActionRecord(
                action_id=actions[0].action_id,
                action_type=actions[0].action_type,
                autonomy_level=actions[0].autonomy_level,
                outcome=actions[0].outcome,
                timestamp=actions[-1].timestamp,
                duration_ms=actions[0].duration_ms,
                input_hash=actions[0].input_hash,
                output_summary=actions[0].output_summary,
            ),
            ActionRecord(
                action_id=actions[-1].action_id,
                action_type=actions[-1].action_type,
                autonomy_level=actions[-1].autonomy_level,
                outcome=actions[-1].outcome,
                timestamp=actions[0].timestamp,
                duration_ms=actions[-1].duration_ms,
                input_hash=actions[-1].input_hash,
                output_summary=actions[-1].output_summary,
            ),
        )

    return SessionTrace(
        session_id=str(uuid4()),
        agent_id="test_agent",
        task_category=task_category,
        session_outcome=session_outcome,
        start_time=base - timedelta(seconds=1),
        end_time=base + timedelta(seconds=num_actions * 10 + 10),
        actions=actions,
    )


class TestTraceValidator:
    def test_well_formed_traces_pass(self):
        validator = TraceValidator()
        traces = [
            _make_trace(
                task_category=cat,
                start_offset_minutes=i * 10,
            )
            for i, cat in enumerate([
                TaskCategory.CODE_REVIEW,
                TaskCategory.RESEARCH,
                TaskCategory.DATA_ANALYSIS,
                TaskCategory.CODE_REVIEW,
                TaskCategory.RESEARCH,
                TaskCategory.DATA_ANALYSIS,
                TaskCategory.CODE_REVIEW,
                TaskCategory.RESEARCH,
                TaskCategory.DATA_ANALYSIS,
                TaskCategory.CODE_REVIEW,
            ])
        ]
        report = validator.validate_batch(traces)
        assert report.total_traces == 10
        assert report.valid_traces == 10
        assert report.validation_rate == 1.0

    def test_out_of_order_timestamps(self):
        validator = TraceValidator()
        trace = _make_trace(num_actions=3, reverse_timestamps=True)
        report = validator.validate_batch([trace])
        assert "timestamps_not_monotonic" in report.errors_by_type

    def test_single_category_low_diversity(self):
        validator = TraceValidator()
        traces = [
            _make_trace(
                task_category=TaskCategory.CODE_REVIEW,
                start_offset_minutes=i * 10,
            )
            for i in range(5)
        ]
        report = validator.validate_batch(traces)
        assert report.diversity_score < 1.0
        assert any("diversity" in w.lower() for w in report.warnings)

    def test_three_categories_full_diversity(self):
        validator = TraceValidator()
        traces = [
            _make_trace(task_category=TaskCategory.CODE_REVIEW),
            _make_trace(task_category=TaskCategory.RESEARCH, start_offset_minutes=10),
            _make_trace(task_category=TaskCategory.DATA_ANALYSIS, start_offset_minutes=20),
        ]
        report = validator.validate_batch(traces)
        assert report.diversity_score == 1.0
        assert not any("diversity" in w.lower() for w in report.warnings)

    def test_duplicate_session_ids(self):
        validator = TraceValidator()
        trace1 = _make_trace()
        # Create a second trace with the same session_id
        trace2 = SessionTrace(
            session_id=trace1.session_id,
            agent_id="test_agent",
            task_category=TaskCategory.RESEARCH,
            session_outcome=SessionOutcome.SUCCESS,
            start_time=trace1.start_time + timedelta(hours=1),
            end_time=trace1.end_time + timedelta(hours=1),
            actions=[
                _make_action(
                    timestamp=trace1.start_time + timedelta(hours=1, seconds=5)
                )
            ],
        )
        report = validator.validate_batch([trace1, trace2])
        assert "duplicate_session_id" in report.errors_by_type

    def test_outcome_inconsistency(self):
        validator = TraceValidator()
        # All actions succeed but session outcome is FAILURE
        trace = _make_trace(
            session_outcome=SessionOutcome.FAILURE,
        )
        report = validator.validate_batch([trace])
        assert "outcome_inconsistency" in report.errors_by_type

    def test_agent_distribution(self):
        validator = TraceValidator()
        traces = [_make_trace(start_offset_minutes=i * 10) for i in range(5)]
        report = validator.validate_batch(traces)
        assert report.agent_distribution == {"test_agent": 5}
