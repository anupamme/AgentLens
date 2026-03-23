"""Full integration tests for the AgentLens pipeline.

These tests run the entire pipeline end-to-end using only mock implementations
(no API calls). The full suite should complete in under 60 seconds.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentlens.aggregation.pipeline import AgentLensPipeline
from agentlens.analysis import AgentAnalyzer, generate_analysis_report
from agentlens.workloads.mock_generator import MockWorkloadGenerator
from agentlens.workloads.runner import WorkloadRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_TYPES = ["code_reviewer", "research_assistant", "task_manager"]
TASKS_PER_AGENT = 10


async def _generate_mock_traces(output_dir: Path) -> int:
    """Generate TASKS_PER_AGENT traces per agent using mock generator."""
    generator = MockWorkloadGenerator(seed=42)
    runner = WorkloadRunner(output_dir=str(output_dir))

    workloads = {
        agent: generator.generate(agent, count=TASKS_PER_AGENT, failure_injection_rate=0.1)
        for agent in AGENT_TYPES
    }

    results = await runner.run_all(workloads, max_concurrent=5)

    total = sum(len(v) for v in results.values())
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipelineMock:
    """End-to-end test: mock workloads → pipeline → analysis → reports."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, tmp_path: Path) -> None:
        """This test should complete in under 60 seconds with no API calls."""
        traces_dir = tmp_path / "traces"
        summaries_dir = tmp_path / "summaries"
        reports_dir = tmp_path / "reports"
        analysis_dir = tmp_path / "analysis"

        # Step 1: Generate mock workloads
        total_traces = await _generate_mock_traces(traces_dir)
        assert total_traces == len(AGENT_TYPES) * TASKS_PER_AGENT

        # Verify trace files were written
        trace_files = list(traces_dir.glob("**/*.jsonl"))
        assert len(trace_files) > 0, "No trace files generated"

        # Step 2: Run aggregation pipeline in mock mode
        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            summaries_dir=str(summaries_dir),
            reports_dir=str(reports_dir),
            use_mock=True,
        )
        report = await pipeline.run()

        # Validate report
        assert report.session_count == len(AGENT_TYPES) * TASKS_PER_AGENT
        assert report.report_id != ""
        assert isinstance(report.mean_autonomous_action_ratio, float)
        assert 0.0 <= report.mean_autonomous_action_ratio <= 1.0
        assert report.high_risk_sessions >= 0

        # Verify summaries were written
        summary_files = list(summaries_dir.glob("*.json"))
        assert len(summary_files) == len(AGENT_TYPES) * TASKS_PER_AGENT

        # Verify report was written
        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1
        report_json = json.loads(report_files[0].read_text())
        assert "session_count" in report_json

        # Step 3: Run analysis
        analyzer = AgentAnalyzer(summaries_dir=str(summaries_dir))
        assert len(analyzer.summaries) == len(AGENT_TYPES) * TASKS_PER_AGENT

        results = analyzer.run_all()
        assert results.metadata["session_count"] == len(AGENT_TYPES) * TASKS_PER_AGENT
        assert len(results.metadata["agent_types"]) == len(AGENT_TYPES)

        # Step 4: Save analysis results
        analyzer.save_results(results, str(analysis_dir))
        results_file = analysis_dir / "results.json"
        assert results_file.exists()
        results_json = json.loads(results_file.read_text())
        assert "autonomy" in results_json
        assert "failures" in results_json
        assert "tools" in results_json
        assert "escalations" in results_json
        assert "oversight_gap" in results_json

        # Step 5: Generate markdown report
        report_md_path = str(analysis_dir / "analysis_report.md")
        generate_analysis_report(results, report_md_path)
        report_md_file = analysis_dir / "analysis_report.md"
        assert report_md_file.exists()
        report_md = report_md_file.read_text()
        assert len(report_md) > 500, "Report too short — likely empty"
        assert "Autonomy" in report_md
        assert "Failure" in report_md

    @pytest.mark.asyncio
    async def test_pipeline_with_diverse_agent_types(self, tmp_path: Path) -> None:
        """Verify each agent type produces traces with correct task categories."""
        from agentlens.aggregation.models import SessionSummary

        traces_dir = tmp_path / "traces"
        summaries_dir = tmp_path / "summaries"
        reports_dir = tmp_path / "reports"

        await _generate_mock_traces(traces_dir)

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            summaries_dir=str(summaries_dir),
            reports_dir=str(reports_dir),
            use_mock=True,
        )
        await pipeline.run()

        summaries = [
            SessionSummary.from_json(p.read_text())
            for p in summaries_dir.glob("*.json")
        ]

        agent_types_seen = {s.agent_type for s in summaries}
        assert agent_types_seen == set(AGENT_TYPES)

        # All summaries should have valid oversight gap scores
        for s in summaries:
            assert 0.0 <= s.oversight_gap_score <= 1.0
            assert s.total_actions > 0

    @pytest.mark.asyncio
    async def test_empty_traces_dir_raises(self, tmp_path: Path) -> None:
        """Pipeline should raise ValueError when no traces are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = AgentLensPipeline(
            traces_dir=str(empty_dir),
            summaries_dir=str(tmp_path / "summaries"),
            reports_dir=str(tmp_path / "reports"),
            use_mock=True,
        )
        with pytest.raises(ValueError, match="No traces found"):
            await pipeline.run()


class TestDatasetExport:
    """Test the HuggingFace dataset preparation script."""

    def test_dataset_export(self, tmp_path: Path) -> None:
        """Test prepare_dataset produces valid JSONL files with correct counts."""
        from scripts.prepare_hf_dataset import prepare_dataset

        traces_dir = tmp_path / "traces"
        summaries_dir = tmp_path / "summaries"
        reports_dir = tmp_path / "reports"
        output_dir = tmp_path / "hf_dataset"

        # Generate mock data
        asyncio.run(_generate_mock_traces(traces_dir))

        # Run pipeline to produce summaries
        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            summaries_dir=str(summaries_dir),
            reports_dir=str(reports_dir),
            use_mock=True,
        )
        asyncio.run(pipeline.run())

        # Run dataset export
        stats = prepare_dataset(
            traces_dir=str(traces_dir),
            summaries_dir=str(summaries_dir),
            reports_dir=str(reports_dir),
            output_dir=str(output_dir),
        )

        # Verify output structure
        assert (output_dir / "README.md").exists()
        assert (output_dir / "dataset_stats.json").exists()
        assert (output_dir / "data" / "traces.jsonl").exists()
        assert (output_dir / "data" / "summaries.jsonl").exists()

        # Verify traces.jsonl has correct number of records
        traces_lines = [
            line for line in (output_dir / "data" / "traces.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(traces_lines) == len(AGENT_TYPES) * TASKS_PER_AGENT

        # Verify summaries.jsonl has correct number of records (≤ traces, after privacy scan)
        summaries_lines = [
            line for line in (output_dir / "data" / "summaries.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(summaries_lines) == len(AGENT_TYPES) * TASKS_PER_AGENT

        # Verify each line is valid JSON
        for line in traces_lines:
            obj = json.loads(line)
            assert "session_id" in obj

        for line in summaries_lines:
            obj = json.loads(line)
            assert "session_id" in obj
            assert "oversight_gap_score" in obj

        # Verify stats
        assert stats["total_traces"] == len(AGENT_TYPES) * TASKS_PER_AGENT
        assert stats["total_summaries"] == len(AGENT_TYPES) * TASKS_PER_AGENT
        assert set(stats["agent_types"].keys()) == set(AGENT_TYPES)

        # Verify dataset card content
        readme = (output_dir / "README.md").read_text()
        assert "AgentLens" in readme
        assert "apache-2.0" in readme
        assert "llm-agents" in readme
        assert "oversight_gap_score" in readme

    def test_dataset_export_empty_dirs(self, tmp_path: Path) -> None:
        """Dataset export should handle empty/missing directories gracefully."""
        from scripts.prepare_hf_dataset import prepare_dataset

        output_dir = tmp_path / "hf_dataset"
        stats = prepare_dataset(
            traces_dir=str(tmp_path / "missing_traces"),
            summaries_dir=str(tmp_path / "missing_summaries"),
            reports_dir=str(tmp_path / "missing_reports"),
            output_dir=str(output_dir),
        )

        assert stats["total_traces"] == 0
        assert stats["total_summaries"] == 0
        assert (output_dir / "README.md").exists()
        assert (output_dir / "data" / "traces.jsonl").exists()
        assert (output_dir / "data" / "summaries.jsonl").exists()
