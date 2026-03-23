"""Workload runner — executes tasks and collects traces."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

from pydantic import BaseModel, Field

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import SessionTrace
from agentlens.sdk.tracer import AgentTracer
from agentlens.sdk.writer import TraceWriter
from agentlens.workloads.failure_injection import FailureInjector, InstrumentedTracer
from agentlens.workloads.generator import (
    AGENT_TYPE_TO_CATEGORY,
    FailureMode,
    TaskConfig,
)


class RunResult(BaseModel):
    """Result of running a single workload task."""

    task_id: str
    agent_type: str
    success: bool
    trace: SessionTrace | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    api_cost_estimate: float = 0.0


# Action sequences per agent type to simulate realistic workflows
_ACTION_SEQUENCES: dict[str, list[tuple[ActionType, str, str]]] = {
    "code_reviewer": [
        (ActionType.READ, "read_source_file", "Reading source code file"),
        (ActionType.SEARCH, "search_patterns", "Searching for code patterns"),
        (ActionType.REASON, "analyze_code", "Analyzing code quality"),
        (ActionType.READ, "read_tests", "Reading test files"),
        (ActionType.REASON, "assess_coverage", "Assessing test coverage"),
        (ActionType.WRITE, "write_review", "Writing review comments"),
        (ActionType.COMMUNICATE, "post_review", "Posting review feedback"),
    ],
    "research_assistant": [
        (ActionType.SEARCH, "web_search", "Searching for information"),
        (ActionType.READ, "read_source", "Reading source material"),
        (ActionType.REASON, "analyze_findings", "Analyzing findings"),
        (ActionType.SEARCH, "deep_search", "Performing deeper search"),
        (ActionType.READ, "read_additional", "Reading additional sources"),
        (ActionType.REASON, "synthesize", "Synthesizing information"),
        (ActionType.WRITE, "write_summary", "Writing research summary"),
    ],
    "task_manager": [
        (ActionType.READ, "read_backlog", "Reading task backlog"),
        (ActionType.REASON, "prioritize", "Prioritizing tasks"),
        (ActionType.SEARCH, "find_dependencies", "Finding task dependencies"),
        (ActionType.REASON, "plan_sprint", "Planning sprint allocation"),
        (ActionType.WRITE, "update_board", "Updating task board"),
        (ActionType.COMMUNICATE, "notify_team", "Notifying team members"),
    ],
    "code_generator": [
        (ActionType.READ, "read_spec", "Reading specification"),
        (ActionType.REASON, "design_solution", "Designing solution"),
        (ActionType.WRITE, "write_code", "Writing code"),
        (ActionType.EXECUTE, "run_tests", "Running tests"),
        (ActionType.REASON, "review_output", "Reviewing test output"),
        (ActionType.WRITE, "refine_code", "Refining implementation"),
        (ActionType.EXECUTE, "final_tests", "Running final tests"),
    ],
    "data_analyst": [
        (ActionType.READ, "load_data", "Loading dataset"),
        (ActionType.REASON, "explore_data", "Exploring data characteristics"),
        (ActionType.EXECUTE, "run_analysis", "Running analysis"),
        (ActionType.REASON, "interpret_results", "Interpreting results"),
        (ActionType.WRITE, "generate_report", "Generating analysis report"),
    ],
    "communicator": [
        (ActionType.READ, "read_context", "Reading discussion context"),
        (ActionType.REASON, "plan_message", "Planning message content"),
        (ActionType.WRITE, "draft_message", "Drafting message"),
        (ActionType.REASON, "review_tone", "Reviewing tone and clarity"),
        (ActionType.COMMUNICATE, "send_message", "Sending message"),
    ],
}

# Autonomy patterns
_AUTONOMY_PATTERNS: dict[str, list[AutonomyLevel]] = {
    "fully_autonomous": [
        AutonomyLevel.FULL_AUTO,
        AutonomyLevel.FULL_AUTO,
        AutonomyLevel.FULL_AUTO,
        AutonomyLevel.AUTO_WITH_AUDIT,
    ],
    "mixed": [
        AutonomyLevel.FULL_AUTO,
        AutonomyLevel.AUTO_WITH_AUDIT,
        AutonomyLevel.HUMAN_CONFIRMED,
        AutonomyLevel.FULL_AUTO,
    ],
    "human_guided": [
        AutonomyLevel.HUMAN_CONFIRMED,
        AutonomyLevel.HUMAN_DRIVEN,
        AutonomyLevel.HUMAN_CONFIRMED,
        AutonomyLevel.AUTO_WITH_AUDIT,
    ],
}


class SimulatedAgent:
    """Drives AgentTracer with dynamic TaskConfig to produce realistic traces."""

    def run(self, task: TaskConfig) -> SessionTrace:
        """Execute a task and return the resulting trace."""
        category = task.task_category
        tracer = AgentTracer(
            agent_type=task.agent_type,
            task_category=category,
            model_used="simulated",
        )

        # If failure mode, wrap the tracer
        active_tracer: AgentTracer | InstrumentedTracer
        if task.injected_failure_mode != FailureMode.NONE:
            active_tracer = FailureInjector.wrap_tracer(tracer, task.injected_failure_mode)
        else:
            active_tracer = tracer

        active_tracer.start_session(task.prompt)

        # Get action sequence for this agent type
        actions = _ACTION_SEQUENCES.get(
            task.agent_type, _ACTION_SEQUENCES["task_manager"]
        )

        # Limit actions based on expected_tool_count
        num_actions = min(len(actions), max(task.expected_tool_count, 1))
        actions = actions[:num_actions]

        # Get autonomy pattern
        autonomy_levels = _AUTONOMY_PATTERNS.get(
            task.expected_autonomy_pattern, _AUTONOMY_PATTERNS["mixed"]
        )

        session_had_failure = False
        for i, (action_type, tool_name, description) in enumerate(actions):
            autonomy = autonomy_levels[i % len(autonomy_levels)]
            raw_input = f"{task.prompt} | step {i + 1}: {description}"

            with active_tracer.action(
                action_type=action_type,
                autonomy_level=autonomy,
                raw_input=raw_input,
                tool_name=tool_name,
                metadata={"step": i + 1, "difficulty": task.difficulty.value},
            ) as ctx:
                # Simulate work
                time.sleep(0.001)
                output = f"Completed: {description} for {task.agent_type} task"
                ctx.set_output_summary(output)

            # Check if failure occurred (from failure injection)
            if tracer._actions and tracer._actions[-1].outcome != ActionOutcome.SUCCESS:
                session_had_failure = True
                # Escalate on safety boundary
                if task.injected_failure_mode == FailureMode.SAFETY_BOUNDARY:
                    active_tracer.record_escalation(
                        reason=EscalationReason.RISK_HIGH,
                        context_summary="Safety boundary triggered during execution",
                    )

        # Determine session outcome
        if session_had_failure:
            if task.injected_failure_mode == FailureMode.PARTIAL_FAILURE:
                outcome = SessionOutcome.PARTIAL
            elif task.injected_failure_mode == FailureMode.SAFETY_BOUNDARY:
                outcome = SessionOutcome.ESCALATED
            elif task.injected_failure_mode == FailureMode.TOOL_TIMEOUT:
                outcome = SessionOutcome.TIMEOUT
            else:
                outcome = SessionOutcome.FAILURE
        else:
            outcome = SessionOutcome.SUCCESS

        trace = active_tracer.end_session(outcome=outcome)
        return trace


class WorkloadRunner:
    """Runs workload tasks and collects traces."""

    def __init__(
        self,
        agents: dict[str, Callable[[TaskConfig], SessionTrace]] | None = None,
        output_dir: str = "./traces",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._default_agent = SimulatedAgent()
        self._agents = agents or {}

    def _get_agent(self, agent_type: str) -> Callable[[TaskConfig], SessionTrace]:
        if agent_type in self._agents:
            return self._agents[agent_type]
        return self._default_agent.run

    async def run_single(self, task: TaskConfig) -> RunResult:
        """Run a single task and save the trace."""
        start = time.monotonic()
        try:
            agent_fn = self._get_agent(task.agent_type)
            trace = await asyncio.get_event_loop().run_in_executor(
                None, agent_fn, task
            )

            # Save trace
            agent_dir = self._output_dir / task.agent_type
            writer = TraceWriter(output_dir=str(agent_dir))
            writer.write_jsonl(trace)

            duration = time.monotonic() - start
            return RunResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=True,
                trace=trace,
                duration_seconds=round(duration, 3),
            )
        except Exception as e:
            duration = time.monotonic() - start
            return RunResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=False,
                error=str(e),
                duration_seconds=round(duration, 3),
            )

    async def run_batch(
        self,
        tasks: list[TaskConfig],
        max_concurrent: int = 3,
    ) -> list[RunResult]:
        """Run a batch of tasks with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one(task: TaskConfig) -> RunResult:
            async with semaphore:
                return await self.run_single(task)

        return await asyncio.gather(*[_run_one(t) for t in tasks])

    async def run_all(
        self,
        workloads: dict[str, list[TaskConfig]],
        max_concurrent: int = 3,
    ) -> dict[str, list[RunResult]]:
        """Run all workloads across agent types."""
        results: dict[str, list[RunResult]] = {}
        all_tasks: list[TaskConfig] = []
        for agent_type, tasks in workloads.items():
            all_tasks.extend(tasks)

        all_results = await self.run_batch(all_tasks, max_concurrent=max_concurrent)

        for result in all_results:
            results.setdefault(result.agent_type, []).append(result)

        return results

    @staticmethod
    def save_run_report(results: dict[str, list[RunResult]], path: str) -> None:
        """Save a summary report of the run."""
        report: dict[str, Any] = {
            "total_tasks": 0,
            "total_success": 0,
            "total_failures": 0,
            "per_agent": {},
        }

        for agent_type, agent_results in results.items():
            successes = sum(1 for r in agent_results if r.success)
            failures = len(agent_results) - successes
            durations = [r.duration_seconds for r in agent_results]

            report["total_tasks"] += len(agent_results)
            report["total_success"] += successes
            report["total_failures"] += failures
            report["per_agent"][agent_type] = {
                "count": len(agent_results),
                "success": successes,
                "failures": failures,
                "success_rate": round(successes / max(len(agent_results), 1), 4),
                "avg_duration_seconds": round(
                    sum(durations) / max(len(durations), 1), 3
                ),
            }

        report["overall_success_rate"] = round(
            report["total_success"] / max(report["total_tasks"], 1), 4
        )

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
