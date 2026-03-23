"""Tests for WorkloadRunner and SimulatedAgent."""

import tempfile
from pathlib import Path

import pytest

from agentlens.schema.enums import ActionOutcome, SessionOutcome
from agentlens.workloads.generator import Difficulty, FailureMode, TaskConfig
from agentlens.workloads.runner import RunResult, SimulatedAgent, WorkloadRunner


class TestSimulatedAgent:
    def test_produces_valid_trace(self):
        agent = SimulatedAgent()
        task = TaskConfig(
            task_id="test_001",
            agent_type="code_reviewer",
            prompt="Review the authentication module for bugs.",
            difficulty=Difficulty.MEDIUM,
            expected_tool_count=3,
        )
        trace = agent.run(task)
        assert trace is not None
        assert trace.session_id
        assert len(trace.actions) >= 1
        assert trace.session_outcome == SessionOutcome.SUCCESS

    def test_timeout_injection(self):
        agent = SimulatedAgent()
        task = TaskConfig(
            task_id="test_timeout",
            agent_type="code_reviewer",
            prompt="Review code with timeout injection.",
            difficulty=Difficulty.EASY,
            expected_tool_count=2,
            injected_failure_mode=FailureMode.TOOL_TIMEOUT,
        )
        trace = agent.run(task)
        # Should have timeout outcomes
        has_timeout = any(
            a.outcome == ActionOutcome.TIMEOUT for a in trace.actions
        )
        assert has_timeout
        assert trace.session_outcome == SessionOutcome.TIMEOUT

    def test_safety_boundary_causes_escalation(self):
        agent = SimulatedAgent()
        task = TaskConfig(
            task_id="test_safety",
            agent_type="research_assistant",
            prompt="Research with safety boundary injection.",
            difficulty=Difficulty.HARD,
            expected_tool_count=3,
            injected_failure_mode=FailureMode.SAFETY_BOUNDARY,
        )
        trace = agent.run(task)
        assert trace.session_outcome == SessionOutcome.ESCALATED
        assert len(trace.escalations) > 0


class TestWorkloadRunner:
    @pytest.mark.asyncio
    async def test_run_single_produces_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = WorkloadRunner(output_dir=tmpdir)
            task = TaskConfig(
                task_id="run_001",
                agent_type="task_manager",
                prompt="Create a sprint plan for the team.",
                difficulty=Difficulty.EASY,
                expected_tool_count=3,
            )
            result = await runner.run_single(task)
            assert isinstance(result, RunResult)
            assert result.success
            assert result.trace is not None
            assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_traces_saved_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = WorkloadRunner(output_dir=tmpdir)
            task = TaskConfig(
                task_id="disk_001",
                agent_type="code_reviewer",
                prompt="Review the API endpoints.",
                difficulty=Difficulty.MEDIUM,
                expected_tool_count=2,
            )
            await runner.run_single(task)

            traces_file = Path(tmpdir) / "code_reviewer" / "traces.jsonl"
            assert traces_file.exists()
            content = traces_file.read_text()
            assert len(content.strip().splitlines()) == 1

    @pytest.mark.asyncio
    async def test_run_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = WorkloadRunner(output_dir=tmpdir)
            tasks = [
                TaskConfig(
                    task_id=f"batch_{i}",
                    agent_type="research_assistant",
                    prompt=f"Research task {i}.",
                    difficulty=Difficulty.EASY,
                    expected_tool_count=2,
                )
                for i in range(5)
            ]
            results = await runner.run_batch(tasks, max_concurrent=2)
            assert len(results) == 5
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_timeout_injection_produces_failure_outcomes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = WorkloadRunner(output_dir=tmpdir)
            task = TaskConfig(
                task_id="timeout_001",
                agent_type="code_reviewer",
                prompt="Review with timeout.",
                difficulty=Difficulty.EASY,
                expected_tool_count=1,
                injected_failure_mode=FailureMode.TOOL_TIMEOUT,
            )
            result = await runner.run_single(task)
            assert result.success  # The run itself succeeds
            assert result.trace is not None
            # But the trace should show timeout outcomes
            has_timeout = any(
                a.outcome == ActionOutcome.TIMEOUT for a in result.trace.actions
            )
            assert has_timeout

    @pytest.mark.asyncio
    async def test_save_run_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = WorkloadRunner(output_dir=tmpdir)
            tasks = [
                TaskConfig(
                    task_id=f"report_{i}",
                    agent_type="task_manager",
                    prompt=f"Task {i}.",
                    difficulty=Difficulty.EASY,
                    expected_tool_count=2,
                )
                for i in range(3)
            ]
            results = await runner.run_batch(tasks)
            results_dict = {"task_manager": results}
            report_path = Path(tmpdir) / "report.json"
            WorkloadRunner.save_run_report(results_dict, str(report_path))
            assert report_path.exists()
            import json
            report = json.loads(report_path.read_text())
            assert report["total_tasks"] == 3
            assert report["total_success"] == 3
