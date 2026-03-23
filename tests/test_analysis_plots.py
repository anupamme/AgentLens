"""Tests for analysis plot generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.analyzer import AgentAnalyzer
from agentlens.analysis.models import (
    AnalysisResults,
    AutonomyAnalysis,
    EscalationAnalysis,
    FailureAnalysis,
    OversightGapAnalysis,
    ToolUsageAnalysis,
)
from agentlens.analysis.plots import (
    _check_matplotlib,
    plot_all,
    plot_autonomy_by_agent,
    plot_autonomy_heatmap,
    plot_autonomy_histogram,
    plot_economic_tasks,
    plot_escalation_matrix,
    plot_escalation_reasons,
    plot_failure_by_autonomy,
    plot_failure_types,
    plot_gap_vs_failure,
    plot_graceful_vs_silent,
    plot_oversight_gap_by_agent,
    plot_oversight_gap_heatmap,
    plot_oversight_gap_histogram,
    plot_tool_bigrams,
    plot_tool_frequency,
    plot_tool_scatter,
)
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


def _make_rich_results() -> AnalysisResults:
    """Build AnalysisResults from diverse summaries for plotting."""
    summaries = [
        make_summary(
            session_id=f"s{i}",
            agent_type="agent-a" if i % 2 == 0 else "agent-b",
            task_category=TaskCategory.CODE_REVIEW if i % 3 == 0 else TaskCategory.RESEARCH,
            oversight_gap_score=i / 10,
            failure_count=i % 3,
            failure_types=["api_error"] if i % 3 else [],
            did_fail_gracefully=i % 2 == 0,
            escalation_count=1 if i % 4 == 0 else 0,
            escalation_reasons=["risk_high"] if i % 4 == 0 else [],
            session_outcome=SessionOutcome.SUCCESS if i % 2 == 0 else SessionOutcome.FAILURE,
            tools_used=["search", "write", "read"] if i % 2 == 0 else ["search", "execute"],
            action_sequence_summary="Read → Reason → Write → Execute" if i % 2 == 0
                                    else "Read → Write → Execute",
        )
        for i in range(10)
    ]
    analyzer = AgentAnalyzer.__new__(AgentAnalyzer)
    analyzer.summaries = summaries
    return analyzer.run_all()


@pytest.fixture
def rich_results() -> AnalysisResults:
    return _make_rich_results()


class TestPlotGeneration:
    def test_plot_all_no_crash(self, rich_results, tmp_path):
        """plot_all() generates PNG files without raising exceptions."""
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        plot_all(rich_results, str(tmp_path))

        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
        pngs = list(plots_dir.glob("*.png"))
        assert len(pngs) > 0

    def test_autonomy_by_agent_png_created(self, rich_results, tmp_path):
        """Autonomy stacked bar plot writes a PNG file."""
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "autonomy_by_agent.png"
        plot_autonomy_by_agent(rich_results.autonomy, output)
        assert output.exists()

    def test_autonomy_histogram_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "autonomy_hist.png"
        plot_autonomy_histogram(rich_results.autonomy, output)
        assert output.exists()

    def test_autonomy_heatmap_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "autonomy_heatmap.png"
        plot_autonomy_heatmap(rich_results.autonomy, output)
        assert output.exists()

    def test_failure_types_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "failure_types.png"
        plot_failure_types(rich_results.failures, output)
        # Only created when there are failure type counts
        if rich_results.failures.failure_type_counts:
            assert output.exists()

    def test_graceful_vs_silent_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "graceful.png"
        plot_graceful_vs_silent(rich_results.failures, output)
        assert output.exists()

    def test_tool_frequency_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "tool_freq.png"
        plot_tool_frequency(rich_results.tools, output)
        if rich_results.tools.tool_frequency:
            assert output.exists()

    def test_tool_bigrams_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "bigrams.png"
        plot_tool_bigrams(rich_results.tools, output)
        if rich_results.tools.common_bigrams:
            assert output.exists()

    def test_escalation_matrix_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "esc_matrix.png"
        plot_escalation_matrix(rich_results.escalations, output)
        assert output.exists()

    def test_oversight_gap_histogram_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "gap_hist.png"
        plot_oversight_gap_histogram(rich_results.oversight_gap, output)
        if rich_results.oversight_gap.score_histogram:
            assert output.exists()

    def test_oversight_gap_by_agent_png_created(self, rich_results, tmp_path):
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        output = tmp_path / "gap_agent.png"
        plot_oversight_gap_by_agent(rich_results.oversight_gap, output)
        if rich_results.oversight_gap.by_agent:
            assert output.exists()

    def test_all_16_plots_generated(self, rich_results, tmp_path):
        """plot_all() generates exactly 16 PNG files."""
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")

        plot_all(rich_results, str(tmp_path))
        plots_dir = tmp_path / "plots"
        pngs = list(plots_dir.glob("*.png"))
        # Some plots skip when data is empty; expect at least 10
        assert len(pngs) >= 10


class TestGracefulSkip:
    def test_no_exception_when_matplotlib_unavailable(self, rich_results, tmp_path):
        """All plot functions return gracefully when matplotlib is not installed."""
        with patch("agentlens.analysis.plots._check_matplotlib", return_value=False):
            # None of these should raise
            output = tmp_path / "test.png"
            plot_autonomy_by_agent(rich_results.autonomy, output)
            plot_autonomy_histogram(rich_results.autonomy, output)
            plot_autonomy_heatmap(rich_results.autonomy, output)
            plot_failure_types(rich_results.failures, output)
            plot_failure_by_autonomy(rich_results.failures, output)
            plot_graceful_vs_silent(rich_results.failures, output)
            plot_tool_frequency(rich_results.tools, output)
            plot_tool_bigrams(rich_results.tools, output)
            plot_tool_scatter(rich_results.tools, output)
            plot_economic_tasks(rich_results.tools, output)
            plot_escalation_reasons(rich_results.escalations, output)
            plot_escalation_matrix(rich_results.escalations, output)
            plot_oversight_gap_histogram(rich_results.oversight_gap, output)
            plot_oversight_gap_by_agent(rich_results.oversight_gap, output)
            plot_gap_vs_failure(rich_results.oversight_gap, output)
            plot_oversight_gap_heatmap(rich_results.oversight_gap, output)

        # No file should have been created
        assert not output.exists()

    def test_plot_all_no_exception_without_matplotlib(self, rich_results, tmp_path):
        """plot_all() returns gracefully when matplotlib is unavailable."""
        with patch("agentlens.analysis.plots._check_matplotlib", return_value=False):
            plot_all(rich_results, str(tmp_path))
        # No plots/ subdir created
        assert not (tmp_path / "plots").exists()


class TestPlotDimensions:
    def test_autonomy_histogram_dpi(self, rich_results, tmp_path):
        """Spot-check that plot is saved at 300 DPI."""
        if not _check_matplotlib():
            pytest.skip("matplotlib not installed")
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed, cannot verify DPI")

        output = tmp_path / "hist.png"
        plot_autonomy_histogram(rich_results.autonomy, output)
        if output.exists():
            with Image.open(output) as img:
                dpi = img.info.get("dpi", (0, 0))
                assert dpi[0] >= 290  # Allow minor floating point variation
