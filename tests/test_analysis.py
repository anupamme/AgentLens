"""Tests for the five-dimensional agent oversight analysis."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.analyzer import AgentAnalyzer
from agentlens.analysis.autonomy import analyze_autonomy
from agentlens.analysis.escalations import analyze_escalations
from agentlens.analysis.failures import analyze_failures
from agentlens.analysis.models import AnalysisResults
from agentlens.analysis.oversight_gap import analyze_oversight_gap
from agentlens.analysis.report import generate_analysis_report
from agentlens.analysis.tools import analyze_tool_usage
from agentlens.schema.enums import SessionOutcome, TaskCategory


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
        "start_time": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        "duration_seconds": 120.0,
        "total_latency_ms": 5000,
        "session_outcome": SessionOutcome.SUCCESS,
        "consequential_action_count": 2,
        "unsupervised_consequential_count": 1,
        "oversight_gap_score": 0.5,
    }
    defaults.update(overrides)
    return SessionSummary(**defaults)


class TestAutonomyAnalysis:
    def test_overall_distribution(self):
        """Known distributions should produce correct means."""
        summaries = [
            make_summary(
                session_id=f"s{i}",
                autonomy_distribution={
                    "fully_autonomous": 0.8,
                    "auto_with_audit": 0.1,
                    "human_confirmed": 0.05,
                    "human_driven": 0.05,
                },
            )
            for i in range(20)
        ]
        result = analyze_autonomy(summaries)

        assert abs(result.overall_distribution["fully_autonomous"] - 0.8) < 0.001
        assert result.mean == pytest.approx(0.8, abs=0.001)
        assert result.median == pytest.approx(0.8, abs=0.001)
        assert result.std == pytest.approx(0.0, abs=0.001)

    def test_high_autonomy_count(self):
        """Sessions with fully_autonomous > 0.8 should be counted."""
        summaries = [
            make_summary(session_id="s1", autonomy_distribution={
                "fully_autonomous": 0.9, "auto_with_audit": 0.1,
                "human_confirmed": 0.0, "human_driven": 0.0,
            }),
            make_summary(session_id="s2", autonomy_distribution={
                "fully_autonomous": 0.5, "auto_with_audit": 0.5,
                "human_confirmed": 0.0, "human_driven": 0.0,
            }),
            make_summary(session_id="s3", autonomy_distribution={
                "fully_autonomous": 0.85, "auto_with_audit": 0.15,
                "human_confirmed": 0.0, "human_driven": 0.0,
            }),
        ]
        result = analyze_autonomy(summaries)

        assert result.high_autonomy_session_count == 2
        assert abs(result.high_autonomy_fraction - 2 / 3) < 0.001

    def test_by_agent_grouping(self):
        """Distribution should be computed per agent type."""
        summaries = [
            make_summary(session_id="s1", agent_type="agent-a", autonomy_distribution={
                "fully_autonomous": 0.9, "auto_with_audit": 0.1,
                "human_confirmed": 0.0, "human_driven": 0.0,
            }),
            make_summary(session_id="s2", agent_type="agent-b", autonomy_distribution={
                "fully_autonomous": 0.2, "auto_with_audit": 0.8,
                "human_confirmed": 0.0, "human_driven": 0.0,
            }),
        ]
        result = analyze_autonomy(summaries)

        assert abs(result.by_agent["agent-a"]["fully_autonomous"] - 0.9) < 0.001
        assert abs(result.by_agent["agent-b"]["fully_autonomous"] - 0.2) < 0.001

    def test_autonomy_ratio_histogram(self):
        """Histogram should contain one entry per session."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(10)]
        result = analyze_autonomy(summaries)

        assert len(result.autonomy_ratio_histogram) == 10

    def test_empty_summaries(self):
        """Empty input should return zero values."""
        result = analyze_autonomy([])

        assert result.high_autonomy_session_count == 0
        assert result.mean == 0.0


class TestFailureAnalysis:
    def test_overall_failure_rate(self):
        """5 failing sessions out of 10 with equal actions."""
        summaries = [
            make_summary(session_id=f"s{i}", total_actions=10, failure_count=2,
                         failure_types=["api_error"], did_fail_gracefully=False)
            for i in range(5)
        ] + [
            make_summary(session_id=f"s{i + 5}", total_actions=10, failure_count=0)
            for i in range(5)
        ]
        result = analyze_failures(summaries)

        # 5*2 failures / (10*10) total actions = 0.1
        assert abs(result.overall_failure_rate - 0.1) < 0.001

    def test_graceful_vs_silent(self):
        """Graceful failure rate from sessions with failures."""
        summaries = [
            make_summary(session_id="s1", failure_count=1, failure_types=["err"],
                         did_fail_gracefully=True),
            make_summary(session_id="s2", failure_count=1, failure_types=["err"],
                         did_fail_gracefully=True),
            make_summary(session_id="s3", failure_count=1, failure_types=["err"],
                         did_fail_gracefully=False),
        ]
        result = analyze_failures(summaries)

        assert abs(result.graceful_failure_rate - 2 / 3) < 0.001
        assert abs(result.silent_failure_rate - 1 / 3) < 0.001

    def test_failure_type_counts(self):
        """Failure type counter accumulates across sessions."""
        summaries = [
            make_summary(session_id="s1", failure_count=1,
                         failure_types=["api_error", "timeout"]),
            make_summary(session_id="s2", failure_count=1,
                         failure_types=["api_error"]),
        ]
        result = analyze_failures(summaries)

        assert result.failure_type_counts["api_error"] == 2
        assert result.failure_type_counts["timeout"] == 1

    def test_no_failures(self):
        """All successful sessions → zero failure rate."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        result = analyze_failures(summaries)

        assert result.overall_failure_rate == 0.0
        assert result.graceful_failure_rate == 1.0

    def test_failure_rate_by_agent(self):
        """Failure rates grouped by agent."""
        summaries = [
            make_summary(session_id="s1", agent_type="good", total_actions=10, failure_count=0),
            make_summary(session_id="s2", agent_type="bad", total_actions=10, failure_count=5,
                         failure_types=["err"], did_fail_gracefully=False),
        ]
        result = analyze_failures(summaries)

        assert result.failure_rate_by_agent["good"] == 0.0
        assert abs(result.failure_rate_by_agent["bad"] - 0.5) < 0.001


class TestToolUsageAnalysis:
    def test_tool_frequency(self):
        """Tool counts accumulate correctly."""
        summaries = [
            make_summary(session_id="s1", tools_used=["search", "write"]),
            make_summary(session_id="s2", tools_used=["search", "read"]),
        ]
        result = analyze_tool_usage(summaries)

        assert result.tool_frequency["search"] == 2
        assert result.tool_frequency["write"] == 1
        assert result.tool_frequency["read"] == 1

    def test_bigram_extraction(self):
        """Bigrams extracted from action_sequence_summary."""
        summaries = [
            make_summary(session_id="s1", action_sequence_summary="A → B → C"),
            make_summary(session_id="s2", action_sequence_summary="A → B → C"),
        ]
        result = analyze_tool_usage(summaries)

        bigram_labels = [bg[0] for bg in result.common_bigrams]
        assert "A → B" in bigram_labels
        assert "B → C" in bigram_labels

        # A→B should appear twice (once per session)
        ab_count = next(count for label, count in result.common_bigrams if label == "A → B")
        assert ab_count == 2

    def test_trigram_extraction(self):
        """Trigrams extracted from sequences of length >= 3."""
        summaries = [
            make_summary(session_id="s1", action_sequence_summary="X → Y → Z"),
        ]
        result = analyze_tool_usage(summaries)

        trigram_labels = [tg[0] for tg in result.common_trigrams]
        assert "X → Y → Z" in trigram_labels

    def test_problematic_tools(self):
        """Tools with success rate < 0.8 flagged as problematic."""
        summaries = [
            make_summary(session_id="s1", tools_used=["bad_tool"],
                         tool_call_count=10, tool_success_rate=0.5),
            make_summary(session_id="s2", tools_used=["good_tool"],
                         tool_call_count=10, tool_success_rate=1.0),
        ]
        result = analyze_tool_usage(summaries)

        assert "bad_tool" in result.problematic_tools
        assert "good_tool" not in result.problematic_tools

    def test_empty_summaries(self):
        result = analyze_tool_usage([])
        assert result.tool_frequency == {}


class TestEscalationAnalysis:
    def test_false_escalation_estimate(self):
        """Escalated + successful = false escalation."""
        summaries = [
            make_summary(session_id="s1", escalation_count=1, escalation_reasons=["risk_high"],
                         session_outcome=SessionOutcome.SUCCESS),
            make_summary(session_id="s2", escalation_count=1, escalation_reasons=["risk_high"],
                         session_outcome=SessionOutcome.FAILURE),
        ]
        result = analyze_escalations(summaries)

        # 1 escalated+succeeded out of 2 escalated sessions = 0.5
        assert abs(result.false_escalation_estimate - 0.5) < 0.001

    def test_missed_escalation_estimate(self):
        """Not escalated + failed = missed escalation."""
        summaries = [
            make_summary(session_id="s1", escalation_count=0,
                         session_outcome=SessionOutcome.FAILURE),
            make_summary(session_id="s2", escalation_count=0,
                         session_outcome=SessionOutcome.SUCCESS),
            make_summary(session_id="s3", escalation_count=0,
                         session_outcome=SessionOutcome.SUCCESS),
        ]
        result = analyze_escalations(summaries)

        # 1 non-escalated failure out of 3 non-escalated = 0.333...
        assert abs(result.missed_escalation_estimate - 1 / 3) < 0.001

    def test_reason_distribution(self):
        """Reason counter accumulates across sessions."""
        summaries = [
            make_summary(session_id="s1", escalation_count=1,
                         escalation_reasons=["confidence_low"]),
            make_summary(session_id="s2", escalation_count=1,
                         escalation_reasons=["confidence_low", "risk_high"]),
        ]
        result = analyze_escalations(summaries)

        assert result.reason_distribution["confidence_low"] == 2
        assert result.reason_distribution["risk_high"] == 1

    def test_overall_escalation_rate(self):
        """Mean of per-session escalation rates."""
        summaries = [
            make_summary(session_id="s1", total_actions=10, escalation_count=2,
                         escalation_reasons=["risk_high", "risk_high"]),
            make_summary(session_id="s2", total_actions=10, escalation_count=0),
        ]
        result = analyze_escalations(summaries)

        # (2/10 + 0/10) / 2 = 0.1
        assert abs(result.overall_escalation_rate - 0.1) < 0.001

    def test_empty_summaries(self):
        result = analyze_escalations([])
        assert result.overall_escalation_rate == 0.0


class TestOversightGapAnalysis:
    def test_correlation_with_failures(self):
        """Linear gap↔failure relationship should produce correlation > 0.9."""
        summaries = []
        for i in range(10):
            gap = i / 9  # 0.0 to 1.0
            failure_count = i  # Proportional to gap
            summaries.append(
                make_summary(
                    session_id=f"s{i}",
                    oversight_gap_score=gap,
                    failure_count=failure_count,
                    total_actions=10,
                )
            )
        result = analyze_oversight_gap(summaries)

        assert result.gap_vs_failure > 0.9

    def test_risk_tiers(self):
        """Sessions correctly classified into risk tiers."""
        summaries = [
            make_summary(session_id="s1", oversight_gap_score=0.1),   # low
            make_summary(session_id="s2", oversight_gap_score=0.5),   # medium
            make_summary(session_id="s3", oversight_gap_score=0.8),   # high
            make_summary(session_id="s4", oversight_gap_score=0.29),  # low
            make_summary(session_id="s5", oversight_gap_score=0.7),   # high (>=0.7)
        ]
        result = analyze_oversight_gap(summaries)

        assert result.low_risk_count == 2
        assert result.medium_risk_count == 1
        assert result.high_risk_count == 2

    def test_top_risk_sessions(self):
        """Top 10 sessions returned sorted by oversight gap score."""
        summaries = [make_summary(session_id=f"s{i}", oversight_gap_score=i / 10)
                     for i in range(15)]
        result = analyze_oversight_gap(summaries)

        assert len(result.top_risk_sessions) == 10
        assert result.top_risk_sessions[0]["oversight_gap_score"] == pytest.approx(1.4, abs=0.01)

    def test_mean_median_std(self):
        """Basic statistics computed correctly."""
        summaries = [
            make_summary(session_id=f"s{i}", oversight_gap_score=float(i) / 4)
            for i in range(5)
        ]
        result = analyze_oversight_gap(summaries)

        assert abs(result.mean_score - 0.5) < 0.01
        assert abs(result.median_score - 0.5) < 0.01

    def test_empty_summaries(self):
        result = analyze_oversight_gap([])
        assert result.mean_score == 0.0
        assert result.high_risk_count == 0


class TestEdgeCases:
    def test_single_summary(self):
        """Single summary should not raise errors."""
        s = make_summary()
        assert analyze_autonomy([s]).high_autonomy_session_count == 0
        assert analyze_failures([s]).overall_failure_rate == 0.0
        assert analyze_tool_usage([s]).avg_unique_tools_per_session == 2.0
        assert analyze_escalations([s]).overall_escalation_rate == 0.0
        assert analyze_oversight_gap([s]).std_score == 0.0

    def test_zero_actions(self):
        """Sessions with zero total_actions should not cause division errors."""
        s = make_summary(total_actions=0, failure_count=0)
        result = analyze_failures([s])
        assert result.overall_failure_rate == 0.0

    def test_all_identical_summaries(self):
        """All-identical summaries (zero variance) should not raise."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        result = analyze_oversight_gap(summaries)
        assert result.std_score == 0.0
        # Pearson on zero variance → 0.0
        assert result.gap_vs_failure == 0.0


class TestAgentAnalyzer:
    def test_run_all_produces_complete_results(self, tmp_path):
        """AgentAnalyzer.run_all() returns AnalysisResults with all five dimensions."""
        # Write summaries to temp dir
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        for s in summaries:
            (tmp_path / f"{s.session_id}.json").write_text(s.to_json())

        analyzer = AgentAnalyzer(summaries_dir=str(tmp_path))
        assert len(analyzer.summaries) == 5

        results = analyzer.run_all()
        assert isinstance(results, AnalysisResults)
        assert results.metadata["session_count"] == 5
        assert isinstance(results.autonomy.mean, float)
        assert isinstance(results.failures.overall_failure_rate, float)
        assert isinstance(results.tools.avg_unique_tools_per_session, float)
        assert isinstance(results.escalations.overall_escalation_rate, float)
        assert isinstance(results.oversight_gap.mean_score, float)

    def test_save_results(self, tmp_path):
        """save_results writes results.json."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(3)]
        (tmp_path / "summaries").mkdir()
        for s in summaries:
            (tmp_path / "summaries" / f"{s.session_id}.json").write_text(s.to_json())

        analyzer = AgentAnalyzer(summaries_dir=str(tmp_path / "summaries"))
        results = analyzer.run_all()

        out_dir = tmp_path / "output"
        analyzer.save_results(results, str(out_dir))

        assert (out_dir / "results.json").exists()

    def test_load_empty_directory(self, tmp_path):
        """Empty summaries dir returns empty list, not an error."""
        analyzer = AgentAnalyzer(summaries_dir=str(tmp_path))
        assert analyzer.summaries == []

    def test_load_jsonl_files(self, tmp_path):
        """AgentAnalyzer loads .jsonl files (one summary per line)."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(3)]
        jsonl_content = "\n".join(s.to_json().replace("\n", " ") for s in summaries)
        (tmp_path / "batch.jsonl").write_text(jsonl_content)

        analyzer = AgentAnalyzer(summaries_dir=str(tmp_path))
        assert len(analyzer.summaries) == 3


class TestReportGeneration:
    def test_report_file_created(self, tmp_path):
        """generate_analysis_report creates a markdown file."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        analyzer = AgentAnalyzer.__new__(AgentAnalyzer)
        analyzer.summaries = summaries
        results = analyzer.run_all()

        report_path = tmp_path / "report.md"
        generate_analysis_report(results, str(report_path))

        assert report_path.exists()

    def test_report_has_all_sections(self, tmp_path):
        """Report contains all 10 expected section headers."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        analyzer = AgentAnalyzer.__new__(AgentAnalyzer)
        analyzer.summaries = summaries
        results = analyzer.run_all()

        report_path = tmp_path / "report.md"
        generate_analysis_report(results, str(report_path))
        content = report_path.read_text()

        assert "## 1. Executive Summary" in content
        assert "## 2. Dataset Overview" in content
        assert "## 3. Autonomy Profiling" in content
        assert "## 4. Failure Taxonomy" in content
        assert "## 5. Tool Usage Patterns" in content
        assert "## 6. Escalation Analysis" in content
        assert "## 7. Oversight Gap Analysis" in content
        assert "## 8. Key Findings" in content
        assert "## 9. Implications for Agent Safety" in content
        assert "## 10. Limitations" in content


class TestSerialization:
    def test_analysis_results_json_roundtrip(self):
        """AnalysisResults survives JSON serialization/deserialization."""
        summaries = [make_summary(session_id=f"s{i}") for i in range(5)]
        analyzer = AgentAnalyzer.__new__(AgentAnalyzer)
        analyzer.summaries = summaries
        results = analyzer.run_all()

        json_str = results.to_json()
        restored = AnalysisResults.from_json(json_str)

        assert restored.metadata["session_count"] == results.metadata["session_count"]
        assert abs(restored.autonomy.mean - results.autonomy.mean) < 0.001
        assert abs(restored.oversight_gap.mean_score - results.oversight_gap.mean_score) < 0.001
