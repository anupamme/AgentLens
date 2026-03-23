"""Tests for MockWorkloadGenerator."""

import pytest

from agentlens.workloads.generator import Difficulty, FailureMode, TaskConfig
from agentlens.workloads.mock_generator import MockWorkloadGenerator


class TestMockWorkloadGenerator:
    def test_generates_correct_count(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("code_reviewer", count=30)
        assert len(tasks) == 30

    def test_generates_correct_count_per_agent_type(self):
        gen = MockWorkloadGenerator(seed=42)
        for agent_type in ["code_reviewer", "research_assistant", "task_manager"]:
            tasks = gen.generate(agent_type, count=25)
            assert len(tasks) == 25
            assert all(t.agent_type == agent_type for t in tasks)

    def test_difficulty_distribution(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("code_reviewer", count=100)
        difficulties = [t.difficulty for t in tasks]
        # Templates have varied difficulties, so we should see multiple values
        assert len(set(difficulties)) >= 2

    def test_all_fields_valid(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("research_assistant", count=20)
        for task in tasks:
            assert task.task_id
            assert task.agent_type == "research_assistant"
            assert task.prompt
            assert isinstance(task.difficulty, Difficulty)
            assert isinstance(task.injected_failure_mode, FailureMode)
            assert task.expected_tool_count >= 0
            assert task.expected_autonomy_pattern in (
                "fully_autonomous", "mixed", "human_guided"
            )

    def test_failure_injection_rate(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("code_reviewer", count=200, failure_injection_rate=0.2)
        injected = sum(1 for t in tasks if t.injected_failure_mode != FailureMode.NONE)
        # With 200 tasks at 20% rate, expect ~40. Allow wide tolerance.
        assert 15 < injected < 80

    def test_no_failure_injection(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("task_manager", count=50, failure_injection_rate=0.0)
        assert all(t.injected_failure_mode == FailureMode.NONE for t in tasks)

    def test_task_category_mapping(self):
        gen = MockWorkloadGenerator(seed=42)
        from agentlens.schema.enums import TaskCategory
        tasks = gen.generate("code_reviewer", count=5)
        assert all(t.task_category == TaskCategory.CODE_REVIEW for t in tasks)

        tasks = gen.generate("research_assistant", count=5)
        assert all(t.task_category == TaskCategory.RESEARCH for t in tasks)

    def test_deterministic_with_seed(self):
        gen1 = MockWorkloadGenerator(seed=123)
        gen2 = MockWorkloadGenerator(seed=123)
        tasks1 = gen1.generate("code_reviewer", count=10)
        tasks2 = gen2.generate("code_reviewer", count=10)
        assert [t.prompt for t in tasks1] == [t.prompt for t in tasks2]

    def test_unknown_agent_type_uses_fallback(self):
        gen = MockWorkloadGenerator(seed=42)
        tasks = gen.generate("unknown_agent", count=5)
        assert len(tasks) == 5
        assert all(t.agent_type == "unknown_agent" for t in tasks)
