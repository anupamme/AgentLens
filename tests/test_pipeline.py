"""Tests for the end-to-end aggregation pipeline."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, SessionTrace
from agentlens.aggregation.models import AggregateReport
from agentlens.aggregation.pipeline import AgentLensPipeline
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


def _write_traces_to_dir(traces: list[SessionTrace], traces_dir: Path) -> None:
    """Write traces as individual JSON files."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        path = traces_dir / f"{trace.session_id}.json"
        path.write_text(trace.to_json())


class TestAgentLensPipeline:
    @pytest.mark.asyncio
    async def test_mock_pipeline_end_to_end(self, tmp_path):
        """Generate 10 traces, run pipeline with use_mock=True, assert report is valid."""
        traces_dir = tmp_path / "traces"
        traces = [
            make_session(
                session_id=f"sess-{i:03d}",
                actions=[
                    make_action(action_id=f"act-{i}-1", action_type=ActionType.READ),
                    make_action(action_id=f"act-{i}-2", action_type=ActionType.WRITE),
                ],
            )
            for i in range(10)
        ]
        _write_traces_to_dir(traces, traces_dir)

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            summaries_dir=str(tmp_path / "summaries"),
            reports_dir=str(tmp_path / "reports"),
            use_mock=True,
        )
        report = await pipeline.run()

        assert isinstance(report, AggregateReport)
        assert report.session_count == 10

    @pytest.mark.asyncio
    async def test_file_output(self, tmp_path):
        """Assert summaries and reports are written to correct directories."""
        traces_dir = tmp_path / "traces"
        summaries_dir = tmp_path / "summaries"
        reports_dir = tmp_path / "reports"

        traces = [
            make_session(
                session_id=f"sess-{i:03d}",
                actions=[make_action(action_id=f"act-{i}")],
            )
            for i in range(3)
        ]
        _write_traces_to_dir(traces, traces_dir)

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            summaries_dir=str(summaries_dir),
            reports_dir=str(reports_dir),
            use_mock=True,
        )
        report = await pipeline.run()

        # Check summaries written
        summary_files = list(summaries_dir.glob("*.json"))
        assert len(summary_files) == 3

        # Check report written
        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1

        # Verify report file content is valid
        report_text = report_files[0].read_text()
        restored = AggregateReport.from_json(report_text)
        assert restored.session_count == 3

    @pytest.mark.asyncio
    async def test_empty_traces_directory(self, tmp_path):
        """Assert graceful error handling for empty traces directory."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            use_mock=True,
        )

        with pytest.raises(ValueError, match="No traces found"):
            await pipeline.run()

    @pytest.mark.asyncio
    async def test_nonexistent_traces_directory(self, tmp_path):
        """Assert graceful error handling for missing traces directory."""
        pipeline = AgentLensPipeline(
            traces_dir=str(tmp_path / "nonexistent"),
            use_mock=True,
        )

        with pytest.raises(ValueError, match="No traces found"):
            await pipeline.run()

    @pytest.mark.asyncio
    async def test_load_traces_jsonl(self, tmp_path):
        """Verify JSONL files are loaded correctly."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        traces = [
            make_session(
                session_id=f"sess-{i:03d}",
                actions=[make_action(action_id=f"act-{i}")],
            )
            for i in range(5)
        ]

        # Write as JSONL
        jsonl_path = traces_dir / "traces.jsonl"
        with open(jsonl_path, "w") as f:
            for trace in traces:
                f.write(trace.model_dump_json() + "\n")

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            use_mock=True,
        )
        loaded = pipeline.load_traces()
        assert len(loaded) == 5

    @pytest.mark.asyncio
    async def test_save_report_to_custom_path(self, tmp_path):
        """Verify save_report writes to the specified path."""
        traces_dir = tmp_path / "traces"
        traces = [
            make_session(actions=[make_action()]),
        ]
        _write_traces_to_dir(traces, traces_dir)

        pipeline = AgentLensPipeline(
            traces_dir=str(traces_dir),
            use_mock=True,
        )
        report = await pipeline.run()

        custom_path = str(tmp_path / "custom" / "my_report.json")
        pipeline.save_report(report, custom_path)
        assert Path(custom_path).exists()

        restored = AggregateReport.from_json(Path(custom_path).read_text())
        assert restored.report_id == report.report_id
