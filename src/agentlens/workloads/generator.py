"""Workload generator — creates TaskConfig objects for synthetic trace generation.

Provides both LLM-based (WorkloadGenerator) and template-based (MockWorkloadGenerator)
generation of task configurations for driving simulated agents.
"""

from __future__ import annotations

import asyncio
import json as _json
import random
import re
import sys
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agentlens.schema.enums import TaskCategory


class Difficulty(str, Enum):
    """Difficulty level for generated tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FailureMode(str, Enum):
    """Injected failure modes for testing agent robustness."""

    NONE = "none"
    TOOL_TIMEOUT = "tool_timeout"
    AMBIGUOUS_INPUT = "ambiguous_input"
    CONFLICTING_CONSTRAINTS = "conflicting_constraints"
    SAFETY_BOUNDARY = "safety_boundary"
    PARTIAL_FAILURE = "partial_failure"


# Map agent_type strings to TaskCategory enums
AGENT_TYPE_TO_CATEGORY: dict[str, TaskCategory] = {
    "code_reviewer": TaskCategory.CODE_REVIEW,
    "code_generator": TaskCategory.CODE_GENERATION,
    "research_assistant": TaskCategory.RESEARCH,
    "data_analyst": TaskCategory.DATA_ANALYSIS,
    "communicator": TaskCategory.COMMUNICATION,
    "task_manager": TaskCategory.SYSTEM_ADMIN,
}

_FAILURE_MODES_WITH_INJECTION = [
    FailureMode.TOOL_TIMEOUT,
    FailureMode.AMBIGUOUS_INPUT,
    FailureMode.CONFLICTING_CONSTRAINTS,
    FailureMode.SAFETY_BOUNDARY,
    FailureMode.PARTIAL_FAILURE,
]


class TaskConfig(BaseModel):
    """Configuration for a single workload task."""

    task_id: str = Field(..., min_length=1)
    agent_type: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    difficulty: Difficulty = Difficulty.MEDIUM
    injected_failure_mode: FailureMode = FailureMode.NONE
    expected_autonomy_pattern: str = "mixed"
    expected_tool_count: int = Field(default=3, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_category(self) -> TaskCategory:
        return AGENT_TYPE_TO_CATEGORY.get(self.agent_type, TaskCategory.OTHER)


GENERATION_PROMPT = """\
You are generating synthetic workload task descriptions for an LLM agent observability framework.

Generate {count} diverse task configurations for a "{agent_type}" agent. Each task should be realistic
and representative of what this type of agent would handle in production.

For each task, provide a JSON object with these fields:
- task_id: unique identifier (string, use format "{agent_type}_NNN")
- prompt: a realistic task prompt the agent would receive (2-4 sentences)
- difficulty: one of "easy", "medium", "hard"
- expected_autonomy_pattern: one of "fully_autonomous", "mixed", "human_guided"
- expected_tool_count: estimated number of tool calls (integer, 1-10)
- metadata: object with optional context (e.g. "domain", "complexity_factors")

Return a JSON array of these objects. No markdown fences, just the JSON array.
"""

AGENT_INSTRUCTIONS: dict[str, str] = {
    "code_reviewer": (
        "Tasks should involve reviewing code for bugs, style issues, security vulnerabilities, "
        "performance problems, and best practices. Vary the programming languages and project types."
    ),
    "research_assistant": (
        "Tasks should involve finding information, synthesizing research, comparing approaches, "
        "and producing summaries. Vary the domains: technical, scientific, business, policy."
    ),
    "task_manager": (
        "Tasks should involve organizing work, prioritizing items, tracking dependencies, "
        "scheduling, and generating status reports. Vary the project types and team sizes."
    ),
    "code_generator": (
        "Tasks should involve writing new code, implementing features, creating tests, "
        "and building components. Vary languages, frameworks, and complexity."
    ),
    "data_analyst": (
        "Tasks should involve analyzing datasets, creating visualizations, finding patterns, "
        "and generating reports. Vary the data types and analysis methods."
    ),
    "communicator": (
        "Tasks should involve drafting messages, summarizing discussions, formatting reports, "
        "and coordinating between teams. Vary the communication channels and audiences."
    ),
}


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


class WorkloadGenerator:
    """LLM-based workload generator using Anthropic API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        aws_region: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for WorkloadGenerator. "
                "Install it with: pip install agentlens[workloads]"
            ) from e
        client: anthropic.AsyncAnthropicBedrock | anthropic.AsyncAnthropic
        if aws_region:
            client = anthropic.AsyncAnthropicBedrock(aws_region=aws_region)
        else:
            client = anthropic.AsyncAnthropic(api_key=api_key)
        self.client = client
        self.model = model

    async def generate(
        self,
        agent_type: str,
        count: int = 200,
        difficulty_distribution: dict[Difficulty, float] | None = None,
        failure_injection_rate: float = 0.2,
    ) -> list[TaskConfig]:
        """Generate task configurations using LLM in batches of 20."""
        if difficulty_distribution is None:
            difficulty_distribution = {
                Difficulty.EASY: 0.3,
                Difficulty.MEDIUM: 0.5,
                Difficulty.HARD: 0.2,
            }

        tasks: list[TaskConfig] = []
        batch_size = 20

        for i, batch_start in enumerate(range(0, count, batch_size)):
            if i > 0:
                await asyncio.sleep(1.0)  # pace between batches
            batch_count = min(batch_size, count - batch_start)
            batch_tasks = await self._generate_batch(agent_type, batch_count, batch_start)
            tasks.extend(batch_tasks)

        # Apply difficulty distribution
        for task in tasks:
            r = random.random()
            cumulative = 0.0
            for diff, weight in difficulty_distribution.items():
                cumulative += weight
                if r <= cumulative:
                    task.difficulty = diff
                    break

        # Apply failure injection
        for task in tasks:
            if random.random() < failure_injection_rate:
                task = self.inject_failure_mode(task)

        return tasks[:count]

    async def _generate_batch(
        self, agent_type: str, count: int, batch_start: int = 0, max_retries: int = 6,
    ) -> list[TaskConfig]:
        """Generate a single batch of tasks via LLM with retry on rate limits."""
        instructions = AGENT_INSTRUCTIONS.get(agent_type, "Generate diverse, realistic tasks.")
        prompt = GENERATION_PROMPT.format(agent_type=agent_type, count=count)
        prompt += f"\n\nAdditional instructions: {instructions}"

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    if attempt < max_retries:
                        delay = min(2 ** attempt, 30) + random.uniform(0, 2)
                        print(
                            f"Rate limited (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s...",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(delay)
                        continue
                raise

        import anthropic as _anthropic
        first_block = response.content[0]
        if not isinstance(first_block, _anthropic.types.TextBlock):
            return []
        raw = _strip_json_fences(first_block.text)
        try:
            items = _json.loads(raw)
        except _json.JSONDecodeError:
            return []

        tasks: list[TaskConfig] = []
        for item in items:
            try:
                task = TaskConfig(
                    task_id=item.get("task_id", f"{agent_type}_{batch_start + len(tasks)}"),
                    agent_type=agent_type,
                    prompt=item.get("prompt", ""),
                    difficulty=Difficulty(item.get("difficulty", "medium")),
                    expected_autonomy_pattern=item.get("expected_autonomy_pattern", "mixed"),
                    expected_tool_count=item.get("expected_tool_count", 3),
                    metadata=item.get("metadata", {}),
                )
                tasks.append(task)
            except (ValueError, ValidationError) as exc:
                print(
                    f"Warning: skipping malformed task item: {exc}",
                    file=sys.stderr,
                )
                continue

        return tasks

    @staticmethod
    def inject_failure_mode(task: TaskConfig) -> TaskConfig:
        """Randomly assign a failure mode to a task."""
        mode = random.choice(_FAILURE_MODES_WITH_INJECTION)
        task.injected_failure_mode = mode
        return task
