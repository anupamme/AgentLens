"""Dimension 3: Tool Usage Patterns."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.models import ToolUsageAnalysis

# Maps tool names to economic task categories
TOOL_TO_ECONOMIC_TASK: dict[str, str] = {
    "search": "information_retrieval",
    "read": "information_retrieval",
    "browse": "information_retrieval",
    "write": "content_generation",
    "generate": "content_generation",
    "code": "content_generation",
    "execute": "task_execution",
    "run": "task_execution",
    "deploy": "task_execution",
    "review": "quality_assurance",
    "test": "quality_assurance",
    "validate": "quality_assurance",
    "email": "communication",
    "message": "communication",
    "notify": "communication",
    "analyze": "data_analysis",
    "query": "data_analysis",
    "compute": "data_analysis",
}


def _parse_sequence(action_sequence_summary: str) -> list[str]:
    """Parse action_sequence_summary into list of steps."""
    return [s.strip() for s in re.split(r"\s*[→\->]+\s*", action_sequence_summary) if s.strip()]


def _extract_ngrams(steps: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of steps using a sliding window."""
    if len(steps) < n:
        return []
    return [tuple(steps[i : i + n]) for i in range(len(steps) - n + 1)]


def analyze_tool_usage(summaries: list[SessionSummary]) -> ToolUsageAnalysis:
    """Analyze tool usage patterns across sessions."""
    if not summaries:
        return ToolUsageAnalysis()

    # Frequency
    tool_counter: Counter[str] = Counter()
    tool_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    for s in summaries:
        for tool in s.tools_used:
            tool_counter[tool] += 1
            tool_by_agent[s.agent_type][tool] += 1

    tool_freq_by_agent = {k: dict(v) for k, v in tool_by_agent.items()}

    # Success rates: distribute tool_success_rate evenly across tools per session
    tool_total_calls: dict[str, float] = {}
    tool_successful_calls: dict[str, float] = {}
    for s in summaries:
        if not s.tools_used:
            continue
        calls_per_tool = s.tool_call_count / len(s.tools_used)
        successes_per_tool = calls_per_tool * s.tool_success_rate
        for tool in s.tools_used:
            tool_total_calls[tool] = tool_total_calls.get(tool, 0.0) + calls_per_tool
            tool_successful_calls[tool] = tool_successful_calls.get(tool, 0.0) + successes_per_tool

    tool_success_rates: dict[str, float] = {}
    for tool in tool_counter:
        total = tool_total_calls.get(tool, 0.0)
        if total > 0:
            tool_success_rates[tool] = round(tool_successful_calls.get(tool, 0.0) / total, 4)
        else:
            tool_success_rates[tool] = 0.0

    # Problematic tools: success_rate < 0.8
    problematic = [t for t, rate in tool_success_rates.items() if rate < 0.8]

    # Sequences: bigrams and trigrams
    bigram_counter: Counter[tuple[str, ...]] = Counter()
    trigram_counter: Counter[tuple[str, ...]] = Counter()
    for s in summaries:
        steps = _parse_sequence(s.action_sequence_summary)
        for bg in _extract_ngrams(steps, 2):
            bigram_counter[bg] += 1
        for tg in _extract_ngrams(steps, 3):
            trigram_counter[tg] += 1

    common_bigrams = [
        (" → ".join(bg), count) for bg, count in bigram_counter.most_common(20)
    ]
    common_trigrams = [
        (" → ".join(tg), count) for tg, count in trigram_counter.most_common(20)
    ]

    # Unique tools per session
    unique_counts = [len(set(s.tools_used)) for s in summaries]
    avg_unique = round(statistics.mean(unique_counts), 4) if unique_counts else 0.0

    # Economic mapping
    economic_counter: Counter[str] = Counter()
    for tool in tool_counter:
        task = TOOL_TO_ECONOMIC_TASK.get(tool.lower(), "other")
        economic_counter[task] += tool_counter[tool]

    return ToolUsageAnalysis(
        tool_frequency=dict(tool_counter.most_common()),
        tool_frequency_by_agent=tool_freq_by_agent,
        tool_success_rates=tool_success_rates,
        problematic_tools=problematic,
        common_bigrams=common_bigrams,
        common_trigrams=common_trigrams,
        avg_unique_tools_per_session=avg_unique,
        tool_to_economic_task=TOOL_TO_ECONOMIC_TASK,
        economic_task_distribution=dict(economic_counter),
    )
