"""Dimension 4: Escalation Analysis."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.models import EscalationAnalysis
from agentlens.schema.enums import SessionOutcome


def analyze_escalations(summaries: list[SessionSummary]) -> EscalationAnalysis:
    """Analyze escalation patterns across sessions."""
    if not summaries:
        return EscalationAnalysis()

    # Overall rate: mean of escalation_count / max(total_actions, 1) per session
    rates = [s.escalation_count / max(s.total_actions, 1) for s in summaries]
    overall_rate = round(statistics.mean(rates), 4)

    # By agent
    by_agent: dict[str, list[SessionSummary]] = defaultdict(list)
    for s in summaries:
        by_agent[s.agent_type].append(s)

    rate_by_agent: dict[str, float] = {}
    for agent, group in sorted(by_agent.items()):
        agent_rates = [s.escalation_count / max(s.total_actions, 1) for s in group]
        rate_by_agent[agent] = round(statistics.mean(agent_rates), 4)

    # Reason distribution
    reason_counter: Counter[str] = Counter()
    reason_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    for s in summaries:
        for reason in s.escalation_reasons:
            reason_counter[reason] += 1
            reason_by_agent[s.agent_type][reason] += 1

    reason_by_agent_clean = {k: dict(v) for k, v in reason_by_agent.items()}

    # False escalation estimate:
    # sessions where escalation_count > 0 AND outcome == SUCCESS / total escalated sessions
    escalated = [s for s in summaries if s.escalation_count > 0]
    if escalated:
        false_esc = sum(1 for s in escalated if s.session_outcome == SessionOutcome.SUCCESS)
        false_rate = round(false_esc / len(escalated), 4)
    else:
        false_rate = 0.0

    # Missed escalation estimate:
    # sessions where escalation_count == 0 AND outcome == FAILURE / total non-escalated sessions
    non_escalated = [s for s in summaries if s.escalation_count == 0]
    if non_escalated:
        missed = sum(1 for s in non_escalated if s.session_outcome == SessionOutcome.FAILURE)
        missed_rate = round(missed / len(non_escalated), 4)
    else:
        missed_rate = 0.0

    # Missed escalation by agent
    missed_by_agent: dict[str, float] = {}
    for agent, group in sorted(by_agent.items()):
        non_esc = [s for s in group if s.escalation_count == 0]
        if non_esc:
            m = sum(1 for s in non_esc if s.session_outcome == SessionOutcome.FAILURE)
            missed_by_agent[agent] = round(m / len(non_esc), 4)
        else:
            missed_by_agent[agent] = 0.0

    # Timing proxy: total_actions * (1 - escalation_count/total_actions)
    # This approximates actions before first escalation since exact timing isn't in SessionSummary
    timing_values = []
    for s in summaries:
        if s.escalation_count > 0:
            proxy = s.total_actions * (1 - s.escalation_count / max(s.total_actions, 1))
            timing_values.append(proxy)

    mean_timing = round(statistics.mean(timing_values), 4) if timing_values else 0.0

    return EscalationAnalysis(
        overall_escalation_rate=overall_rate,
        escalation_rate_by_agent=rate_by_agent,
        reason_distribution=dict(reason_counter),
        reason_by_agent=reason_by_agent_clean,
        false_escalation_estimate=false_rate,
        missed_escalation_estimate=missed_rate,
        missed_escalation_by_agent=missed_by_agent,
        mean_actions_before_first_escalation=mean_timing,
    )
