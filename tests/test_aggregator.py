"""Tests for the cross-session aggregator."""

import pytest

from agentlens.schema.enums import SessionOutcome, TaskCategory
from agentlens.aggregation.models import AggregateReport, SessionSummary
from agentlens.aggregation.aggregator import MockAggregator, compute_statistics


def make_summary(**overrides):
    defaults = {
        "session_id": "sess-001",
        "agent_type": "test-agent",
        "task_category": TaskCategory.CODE_REVIEW,
        "task_abstract": "Agent performed 5 actions on a code review task",
        "action_sequence_summary": "Read → Reason → Write",
        "total_actions": 5,
        "autonomy_distribution": {
            "fully_autonomous": 0.6,
            "auto_with_audit": 0.2,
            "human_confirmed": 0.1,
            "human_driven": 0.1,
        },
        "tools_used": ["search", "write"],
        "tool_call_count": 3,
        "tool_success_rate": 1.0,
        "failure_count": 0,
        "failure_types": [],
        "escalation_count": 0,
        "escalation_reasons": [],
        "did_fail_gracefully": True,
        "duration_seconds": 120.0,
        "total_latency_ms": 5000,
        "session_outcome": SessionOutcome.SUCCESS,
        "consequential_action_count": 2,
        "unsupervised_consequential_count": 1,
        "oversight_gap_score": 0.5,
    }
    defaults.update(overrides)
    return SessionSummary(**defaults)


class TestComputeStatistics:
    def test_basic_statistics(self):
        """Feed known summaries, verify means and distributions."""
        summaries = [
            make_summary(
                session_id="s1",
                agent_type="agent-a",
                task_category=TaskCategory.CODE_REVIEW,
                total_actions=10,
                oversight_gap_score=0.4,
                duration_seconds=100.0,
                session_outcome=SessionOutcome.SUCCESS,
            ),
            make_summary(
                session_id="s2",
                agent_type="agent-b",
                task_category=TaskCategory.RESEARCH,
                total_actions=20,
                oversight_gap_score=0.6,
                duration_seconds=200.0,
                session_outcome=SessionOutcome.PARTIAL,
            ),
        ]
        stats = compute_statistics(summaries)

        assert stats["session_count"] == 2
        assert stats["task_category_distribution"] == {"code_review": 1, "research": 1}
        assert stats["agent_type_distribution"] == {"agent-a": 1, "agent-b": 1}
        assert abs(stats["mean_oversight_gap_score"] - 0.5) < 0.01
        assert abs(stats["mean_duration_seconds"] - 150.0) < 0.01
        assert abs(stats["mean_actions_per_session"] - 15.0) < 0.01
        assert stats["outcome_distribution"] == {"success": 1, "partial": 1}

    def test_single_session(self):
        """Edge case: single summary should not cause errors."""
        summaries = [make_summary()]
        stats = compute_statistics(summaries)

        assert stats["session_count"] == 1
        assert abs(stats["mean_oversight_gap_score"] - 0.5) < 0.01

    def test_empty_summaries(self):
        """Edge case: empty list should return zero/empty stats."""
        stats = compute_statistics([])

        assert stats["session_count"] == 0
        assert stats["task_category_distribution"] == {}
        assert stats["mean_oversight_gap_score"] == 0.0
        assert stats["high_risk_sessions"] == 0

    def test_all_failures(self):
        """Edge case: all sessions failed."""
        summaries = [
            make_summary(
                session_id=f"s{i}",
                session_outcome=SessionOutcome.FAILURE,
                failure_count=3,
                failure_types=["api_error"],
                did_fail_gracefully=False,
            )
            for i in range(5)
        ]
        stats = compute_statistics(summaries)

        assert stats["outcome_distribution"] == {"failure": 5}
        assert stats["graceful_failure_rate"] == 0.0

    def test_all_autonomous(self):
        """Edge case: all actions fully autonomous."""
        summaries = [
            make_summary(
                session_id=f"s{i}",
                autonomy_distribution={
                    "fully_autonomous": 1.0,
                    "auto_with_audit": 0.0,
                    "human_confirmed": 0.0,
                    "human_driven": 0.0,
                },
            )
            for i in range(3)
        ]
        stats = compute_statistics(summaries)

        assert abs(stats["mean_autonomous_action_ratio"] - 1.0) < 0.001
        assert abs(stats["autonomy_histogram"]["fully_autonomous"] - 1.0) < 0.001

    def test_zero_escalations(self):
        """Edge case: no escalations at all."""
        summaries = [
            make_summary(session_id=f"s{i}", escalation_count=0, escalation_reasons=[])
            for i in range(3)
        ]
        stats = compute_statistics(summaries)

        assert stats["mean_escalation_rate"] == 0.0
        assert stats["escalation_reason_distribution"] == {}

    def test_high_risk_session_counting(self):
        """Create summaries with known oversight gaps, verify count."""
        summaries = [
            make_summary(session_id="s1", oversight_gap_score=0.3),  # not high risk
            make_summary(session_id="s2", oversight_gap_score=0.71),  # high risk
            make_summary(session_id="s3", oversight_gap_score=0.9),   # high risk
            make_summary(session_id="s4", oversight_gap_score=0.7),   # exactly 0.7 → NOT high risk
            make_summary(session_id="s5", oversight_gap_score=1.0),   # high risk
        ]
        stats = compute_statistics(summaries)

        assert stats["high_risk_sessions"] == 3

    def test_oversight_gap_by_agent(self):
        """Verify oversight gap is grouped correctly by agent type."""
        summaries = [
            make_summary(session_id="s1", agent_type="agent-a", oversight_gap_score=0.2),
            make_summary(session_id="s2", agent_type="agent-a", oversight_gap_score=0.4),
            make_summary(session_id="s3", agent_type="agent-b", oversight_gap_score=0.8),
        ]
        stats = compute_statistics(summaries)

        assert abs(stats["oversight_gap_by_agent"]["agent-a"] - 0.3) < 0.01
        assert abs(stats["oversight_gap_by_agent"]["agent-b"] - 0.8) < 0.01

    def test_failure_rate_by_agent(self):
        """Verify failure rates grouped by agent type."""
        summaries = [
            make_summary(
                session_id="s1", agent_type="good-agent",
                total_actions=10, failure_count=0,
            ),
            make_summary(
                session_id="s2", agent_type="bad-agent",
                total_actions=10, failure_count=5,
                failure_types=["error"],
                did_fail_gracefully=False,
            ),
        ]
        stats = compute_statistics(summaries)

        assert stats["failure_rate_by_agent"]["good-agent"] == 0.0
        assert abs(stats["failure_rate_by_agent"]["bad-agent"] - 0.5) < 0.01


class TestMockAggregator:
    @pytest.mark.asyncio
    async def test_end_to_end(self):
        """Run MockAggregator and assert valid AggregateReport."""
        summaries = [
            make_summary(session_id=f"s{i}") for i in range(20)
        ]
        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        assert isinstance(report, AggregateReport)
        assert report.session_count == 20
        assert report.generated_at is not None
        assert report.report_id  # non-empty

    @pytest.mark.asyncio
    async def test_executive_summary_populated(self):
        """Executive summary should be a non-empty string."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        assert len(report.executive_summary) > 0
        assert "5 sessions" in report.executive_summary

    @pytest.mark.asyncio
    async def test_concerns_for_high_risk(self):
        """When high_risk_sessions > 0, concerns should be populated."""
        summaries = [
            make_summary(session_id="s1", oversight_gap_score=0.9),
            make_summary(session_id="s2", oversight_gap_score=0.8),
        ]
        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        assert report.high_risk_sessions == 2
        assert len(report.concerns) > 0
        assert any("oversight gap" in c for c in report.concerns)

    @pytest.mark.asyncio
    async def test_report_serialization(self):
        """Verify report can be serialized and deserialized."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(3)]
        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        json_str = report.to_json()
        restored = AggregateReport.from_json(json_str)
        assert restored.session_count == report.session_count
        assert restored.report_id == report.report_id
