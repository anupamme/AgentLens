"""Dimension 5: Oversight Gap Score analysis."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.models import OversightGapAnalysis, _pearson_r


def _risk_tier(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "medium"
    else:
        return "high"


def analyze_oversight_gap(summaries: list[SessionSummary]) -> OversightGapAnalysis:
    """Analyze oversight gap scores across sessions."""
    if not summaries:
        return OversightGapAnalysis()

    scores = [s.oversight_gap_score for s in summaries]

    mean_score = round(statistics.mean(scores), 4)
    median_score = round(statistics.median(scores), 4)
    std_score = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0

    # By agent
    by_agent: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        by_agent[s.agent_type].append(s.oversight_gap_score)

    agent_means: dict[str, float] = {
        agent: round(statistics.mean(vals), 4) for agent, vals in sorted(by_agent.items())
    }

    # By task category
    by_task: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        by_task[s.task_category.value].append(s.oversight_gap_score)

    task_means: dict[str, float] = {
        task: round(statistics.mean(vals), 4) for task, vals in sorted(by_task.items())
    }

    # Risk tiers
    low = sum(1 for sc in scores if _risk_tier(sc) == "low")
    medium = sum(1 for sc in scores if _risk_tier(sc) == "medium")
    high = sum(1 for sc in scores if _risk_tier(sc) == "high")

    # Risk tier by agent
    risk_by_agent: dict[str, dict[str, int]] = {}
    for agent, vals in sorted(by_agent.items()):
        risk_by_agent[agent] = {
            "low": sum(1 for v in vals if _risk_tier(v) == "low"),
            "medium": sum(1 for v in vals if _risk_tier(v) == "medium"),
            "high": sum(1 for v in vals if _risk_tier(v) == "high"),
        }

    # Correlations
    failure_rates = [s.failure_count / max(s.total_actions, 1) for s in summaries]
    durations = [s.duration_seconds for s in summaries]
    action_counts = [float(s.total_actions) for s in summaries]

    gap_vs_failure = round(_pearson_r(scores, failure_rates), 4)
    gap_vs_duration = round(_pearson_r(scores, durations), 4)
    gap_vs_action_count = round(_pearson_r(scores, action_counts), 4)

    # Top risk sessions (top 10 by score)
    sorted_sessions = sorted(summaries, key=lambda s: s.oversight_gap_score, reverse=True)
    top_risk: list[dict[str, Any]] = []
    for s in sorted_sessions[:10]:
        top_risk.append({
            "session_id": s.session_id,
            "agent_type": s.agent_type,
            "task_abstract": s.task_abstract,
            "oversight_gap_score": s.oversight_gap_score,
            "consequential_action_count": s.consequential_action_count,
            "unsupervised_consequential_count": s.unsupervised_consequential_count,
        })

    return OversightGapAnalysis(
        mean_score=mean_score,
        median_score=median_score,
        std_score=std_score,
        score_histogram=scores,
        by_agent=agent_means,
        by_task_category=task_means,
        low_risk_count=low,
        medium_risk_count=medium,
        high_risk_count=high,
        risk_tier_by_agent=risk_by_agent,
        gap_vs_failure=gap_vs_failure,
        gap_vs_duration=gap_vs_duration,
        gap_vs_action_count=gap_vs_action_count,
        top_risk_sessions=top_risk,
    )
