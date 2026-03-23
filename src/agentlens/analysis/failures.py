"""Dimension 2: Failure Taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.models import FailureAnalysis


def _dominant_autonomy_level(s: SessionSummary) -> str:
    """Return the autonomy level with the highest fraction for a session."""
    dist = s.autonomy_distribution
    if not dist:
        return "unknown"
    return max(dist, key=lambda k: dist[k])


def analyze_failures(summaries: list[SessionSummary]) -> FailureAnalysis:
    """Analyze failure patterns across sessions."""
    if not summaries:
        return FailureAnalysis()

    total_failures = sum(s.failure_count for s in summaries)
    total_actions = sum(s.total_actions for s in summaries)
    overall_rate = round(total_failures / max(total_actions, 1), 4)

    # By agent
    by_agent: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        by_agent[s.agent_type].append(s)

    rate_by_agent: dict[str, float] = {}
    for agent, group in sorted(by_agent.items()):
        af = sum(s.failure_count for s in group)
        aa = sum(s.total_actions for s in group)
        rate_by_agent[agent] = round(af / max(aa, 1), 4)

    # By task
    by_task: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        by_task[s.task_category.value].append(s)

    rate_by_task: dict[str, float] = {}
    for task, group in sorted(by_task.items()):
        tf = sum(s.failure_count for s in group)
        ta = sum(s.total_actions for s in group)
        rate_by_task[task] = round(tf / max(ta, 1), 4)

    # Failure type counts
    type_counter: Counter[str] = Counter()
    type_by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in summaries:
        for ft in s.failure_types:
            type_counter[ft] += 1
            type_by_agent[s.agent_type][ft] += 1

    # Convert nested defaultdicts to regular dicts
    type_by_agent_clean = {k: dict(v) for k, v in type_by_agent.items()}

    # Graceful vs silent
    sessions_with_failures = [s for s in summaries if s.failure_count > 0]
    if sessions_with_failures:
        graceful_count = sum(1 for s in sessions_with_failures if s.did_fail_gracefully)
        graceful_rate = round(graceful_count / len(sessions_with_failures), 4)
        silent_rate = round(1.0 - graceful_rate, 4)
    else:
        graceful_rate = 1.0
        silent_rate = 0.0

    # Graceful by agent
    graceful_by_agent: dict[str, float] = {}
    for agent, group in sorted(by_agent.items()):
        failed = [s for s in group if s.failure_count > 0]
        if failed:
            gc = sum(1 for s in failed if s.did_fail_gracefully)
            graceful_by_agent[agent] = round(gc / len(failed), 4)
        else:
            graceful_by_agent[agent] = 1.0

    # Failure rate by autonomy level bucket
    by_autonomy: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        level = _dominant_autonomy_level(s)
        by_autonomy[level].append(s)

    rate_by_autonomy: dict[str, float] = {}
    for level, group in sorted(by_autonomy.items()):
        lf = sum(s.failure_count for s in group)
        la = sum(s.total_actions for s in group)
        rate_by_autonomy[level] = round(lf / max(la, 1), 4)

    # Autonomous vs supervised failure ratio
    auto_rate = rate_by_autonomy.get("fully_autonomous", 0.0)
    supervised_rate = rate_by_autonomy.get("human_confirmed", 0.0) or rate_by_autonomy.get(
        "human_driven", 0.0
    )
    if supervised_rate > 0:
        ratio = round(auto_rate / supervised_rate, 4)
    else:
        ratio = 0.0 if auto_rate == 0 else float("inf")

    return FailureAnalysis(
        overall_failure_rate=overall_rate,
        failure_rate_by_agent=rate_by_agent,
        failure_rate_by_task=rate_by_task,
        failure_type_counts=dict(type_counter),
        failure_type_by_agent=type_by_agent_clean,
        graceful_failure_rate=graceful_rate,
        silent_failure_rate=silent_rate,
        graceful_by_agent=graceful_by_agent,
        failure_rate_by_autonomy_level=rate_by_autonomy,
        autonomous_vs_supervised_failure_ratio=ratio,
    )
