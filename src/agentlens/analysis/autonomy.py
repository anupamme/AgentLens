"""Dimension 1: Autonomy Profiling."""

from __future__ import annotations

import statistics
from collections import defaultdict

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.models import AutonomyAnalysis


def analyze_autonomy(summaries: list[SessionSummary]) -> AutonomyAnalysis:
    """Analyze autonomy distribution across sessions."""
    if not summaries:
        return AutonomyAnalysis()

    # Overall distribution: element-wise mean of autonomy_distribution dicts
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(s.autonomy_distribution.keys())

    overall: dict[str, float] = {}
    for key in sorted(all_keys):
        vals = [s.autonomy_distribution.get(key, 0.0) for s in summaries]
        overall[key] = round(statistics.mean(vals), 4)

    # By agent
    by_agent: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        by_agent[s.agent_type].append(s)

    agent_distributions: dict[str, dict[str, float]] = {}
    for agent, group in sorted(by_agent.items()):
        dist: dict[str, float] = {}
        for key in sorted(all_keys):
            vals = [s.autonomy_distribution.get(key, 0.0) for s in group]
            dist[key] = round(statistics.mean(vals), 4)
        agent_distributions[agent] = dist

    # By task category
    by_task: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        by_task[s.task_category.value].append(s)

    task_distributions: dict[str, dict[str, float]] = {}
    for task, group in sorted(by_task.items()):
        dist = {}
        for key in sorted(all_keys):
            vals = [s.autonomy_distribution.get(key, 0.0) for s in group]
            dist[key] = round(statistics.mean(vals), 4)
        task_distributions[task] = dist

    # Autonomy ratio histogram: fully_autonomous per session
    ratios = [s.autonomy_distribution.get("fully_autonomous", 0.0) for s in summaries]

    # Stats
    mean_val = round(statistics.mean(ratios), 4)
    std_val = round(statistics.stdev(ratios), 4) if len(ratios) > 1 else 0.0
    median_val = round(statistics.median(ratios), 4)

    # High autonomy: sessions where fully_autonomous > 0.8
    high_autonomy = [
        s for s in summaries if s.autonomy_distribution.get("fully_autonomous", 0.0) > 0.8
    ]
    high_count = len(high_autonomy)
    high_fraction = round(high_count / len(summaries), 4)

    high_by_agent: dict[str, int] = defaultdict(int)
    for s in high_autonomy:
        high_by_agent[s.agent_type] += 1

    return AutonomyAnalysis(
        overall_distribution=overall,
        by_agent=agent_distributions,
        by_task_category=task_distributions,
        autonomy_ratio_histogram=ratios,
        mean=mean_val,
        std=std_val,
        median=median_val,
        high_autonomy_session_count=high_count,
        high_autonomy_fraction=high_fraction,
        high_autonomy_by_agent=dict(high_by_agent),
    )
