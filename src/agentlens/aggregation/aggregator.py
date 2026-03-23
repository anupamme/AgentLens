"""Cross-Session Aggregator — produces AggregateReport from SessionSummary objects.

Aggregates multiple SessionSummary objects into an AggregateReport using a
combination of statistical computation and optional LLM-generated narrative.

Usage:
    aggregator = SessionAggregator(api_key="...")
    report = await aggregator.aggregate(summaries)
"""

from __future__ import annotations

import asyncio
import random
import statistics
import sys
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone

from agentlens.aggregation.models import AggregateReport, SessionSummary
from agentlens.aggregation.summarizer import _strip_markdown_fences


def compute_statistics(summaries: list[SessionSummary]) -> dict:
    """Pure Python statistical aggregation. No LLM needed."""
    n = len(summaries)
    if n == 0:
        return _empty_stats()

    # Task and agent distributions
    task_category_distribution = dict(Counter(
        s.task_category.value for s in summaries
    ))
    agent_type_distribution = dict(Counter(s.agent_type for s in summaries))

    # Autonomy overview
    autonomous_ratios = [
        s.autonomy_distribution.get("fully_autonomous", 0.0) for s in summaries
    ]
    mean_autonomous_action_ratio = _safe_mean(autonomous_ratios)

    # Element-wise mean of autonomy distributions
    all_keys = set()
    for s in summaries:
        all_keys.update(s.autonomy_distribution.keys())
    autonomy_histogram = {}
    for key in sorted(all_keys):
        values = [s.autonomy_distribution.get(key, 0.0) for s in summaries]
        autonomy_histogram[key] = round(_safe_mean(values), 4)

    # Tool usage ranking
    tool_counts: Counter[str] = Counter()
    tool_successes: Counter[str] = Counter()
    tool_totals: Counter[str] = Counter()
    for s in summaries:
        for tool in s.tools_used:
            tool_counts[tool] += 1
        # Distribute success rate across tools proportionally
        if s.tools_used and s.tool_call_count > 0:
            calls_per_tool = s.tool_call_count / len(s.tools_used)
            for tool in s.tools_used:
                tool_totals[tool] += calls_per_tool
                tool_successes[tool] += calls_per_tool * s.tool_success_rate

    tool_usage_ranking = []
    for tool, count in tool_counts.most_common():
        success_rate = (
            tool_successes[tool] / tool_totals[tool]
            if tool_totals[tool] > 0 else 1.0
        )
        tool_usage_ranking.append({
            "tool": tool,
            "count": count,
            "success_rate": round(success_rate, 4),
        })

    # Most common action sequences (top 10)
    sequence_counts = Counter(s.action_sequence_summary for s in summaries)
    most_common_tool_sequences = [
        seq for seq, _ in sequence_counts.most_common(10)
    ]

    # Failure taxonomy
    failure_rate_by_agent: dict[str, float] = {}
    agent_groups: dict[str, list[SessionSummary]] = {}
    for s in summaries:
        agent_groups.setdefault(s.agent_type, []).append(s)
    for agent, group in agent_groups.items():
        total_actions = sum(s.total_actions for s in group)
        total_failures = sum(s.failure_count for s in group)
        failure_rate_by_agent[agent] = round(
            total_failures / max(total_actions, 1), 4
        )

    failure_type_counter: Counter[str] = Counter()
    for s in summaries:
        failure_type_counter.update(s.failure_types)
    failure_type_distribution = dict(failure_type_counter)

    sessions_with_failures = [s for s in summaries if s.failure_count > 0]
    graceful_count = sum(1 for s in sessions_with_failures if s.did_fail_gracefully)
    graceful_failure_rate = (
        graceful_count / len(sessions_with_failures)
        if sessions_with_failures else 1.0
    )

    # Escalation analysis
    escalation_rates = [
        s.escalation_count / max(s.total_actions, 1) for s in summaries
    ]
    mean_escalation_rate = round(_safe_mean(escalation_rates), 4)

    escalation_reason_counter: Counter[str] = Counter()
    for s in summaries:
        escalation_reason_counter.update(s.escalation_reasons)
    escalation_reason_distribution = dict(escalation_reason_counter)

    # Oversight gap
    oversight_gaps = [s.oversight_gap_score for s in summaries]
    mean_oversight_gap_score = round(_safe_mean(oversight_gaps), 4)

    oversight_gap_by_agent = {}
    for agent, group in agent_groups.items():
        gaps = [s.oversight_gap_score for s in group]
        oversight_gap_by_agent[agent] = round(_safe_mean(gaps), 4)

    category_groups: dict[str, list[SessionSummary]] = {}
    for s in summaries:
        category_groups.setdefault(s.task_category.value, []).append(s)
    oversight_gap_by_task_category = {}
    for cat, group in category_groups.items():
        gaps = [s.oversight_gap_score for s in group]
        oversight_gap_by_task_category[cat] = round(_safe_mean(gaps), 4)

    high_risk_sessions = sum(1 for s in summaries if s.oversight_gap_score > 0.7)

    # Performance
    durations = [s.duration_seconds for s in summaries]
    mean_duration_seconds = round(_safe_mean(durations), 2)
    actions_counts = [s.total_actions for s in summaries]
    mean_actions_per_session = round(_safe_mean(actions_counts), 2)

    outcome_distribution = dict(Counter(
        s.session_outcome.value for s in summaries
    ))

    # Time range from actual session timestamps
    time_range = {
        "start": min(s.start_time for s in summaries).isoformat(),
        "end": max(s.end_time for s in summaries).isoformat(),
    }

    return {
        "session_count": n,
        "time_range": time_range,
        "task_category_distribution": task_category_distribution,
        "agent_type_distribution": agent_type_distribution,
        "mean_autonomous_action_ratio": round(mean_autonomous_action_ratio, 4),
        "autonomy_histogram": autonomy_histogram,
        "tool_usage_ranking": tool_usage_ranking,
        "most_common_tool_sequences": most_common_tool_sequences,
        "failure_rate_by_agent": failure_rate_by_agent,
        "failure_type_distribution": failure_type_distribution,
        "graceful_failure_rate": round(graceful_failure_rate, 4),
        "mean_escalation_rate": mean_escalation_rate,
        "escalation_reason_distribution": escalation_reason_distribution,
        "mean_oversight_gap_score": mean_oversight_gap_score,
        "oversight_gap_by_agent": oversight_gap_by_agent,
        "oversight_gap_by_task_category": oversight_gap_by_task_category,
        "high_risk_sessions": high_risk_sessions,
        "mean_duration_seconds": mean_duration_seconds,
        "mean_actions_per_session": mean_actions_per_session,
        "outcome_distribution": outcome_distribution,
    }


def _safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return statistics.mean(values)


def _empty_stats() -> dict:
    """Return an empty statistics dictionary for zero summaries."""
    return {
        "session_count": 0,
        "time_range": {},
        "task_category_distribution": {},
        "agent_type_distribution": {},
        "mean_autonomous_action_ratio": 0.0,
        "autonomy_histogram": {},
        "tool_usage_ranking": [],
        "most_common_tool_sequences": [],
        "failure_rate_by_agent": {},
        "failure_type_distribution": {},
        "graceful_failure_rate": 1.0,
        "mean_escalation_rate": 0.0,
        "escalation_reason_distribution": {},
        "mean_oversight_gap_score": 0.0,
        "oversight_gap_by_agent": {},
        "oversight_gap_by_task_category": {},
        "high_risk_sessions": 0,
        "mean_duration_seconds": 0.0,
        "mean_actions_per_session": 0.0,
        "outcome_distribution": {},
    }


class BaseAggregator(ABC):
    """Abstract base for session aggregators."""

    @abstractmethod
    async def aggregate(self, summaries: list[SessionSummary]) -> AggregateReport:
        ...


class MockAggregator(BaseAggregator):
    """Deterministic aggregator for testing. No LLM calls."""

    async def aggregate(self, summaries: list[SessionSummary]) -> AggregateReport:
        stats = compute_statistics(summaries)
        n = stats["session_count"]

        # Deterministic narrative
        executive_summary = (
            f"Aggregate report covering {n} sessions. "
            f"Mean oversight gap score: {stats['mean_oversight_gap_score']}. "
            f"High risk sessions: {stats['high_risk_sessions']}."
        )

        key_findings = []
        if n > 0:
            key_findings.append(
                f"Average success rate across agents: "
                f"{_format_outcome_rate(stats['outcome_distribution'], n)}"
            )
            key_findings.append(
                f"Mean autonomous action ratio: "
                f"{stats['mean_autonomous_action_ratio']:.1%}"
            )
            if stats["high_risk_sessions"] > 0:
                key_findings.append(
                    f"{stats['high_risk_sessions']} session(s) flagged as high risk "
                    f"(oversight gap > 0.7)"
                )

        concerns = []
        if stats["high_risk_sessions"] > 0:
            concerns.append(
                f"{stats['high_risk_sessions']} session(s) had oversight gap > 0.7, "
                f"indicating significant unsupervised consequential actions"
            )
        if stats["graceful_failure_rate"] < 1.0:
            concerns.append(
                "Not all failures triggered appropriate escalations"
            )

        return AggregateReport(
            generated_at=datetime.now(timezone.utc),
            executive_summary=executive_summary,
            key_findings=key_findings,
            concerns=concerns,
            **stats,
        )


def _format_outcome_rate(outcome_dist: dict[str, int], total: int) -> str:
    """Format success rate from outcome distribution."""
    successes = outcome_dist.get("success", 0)
    return f"{successes / total:.1%}" if total > 0 else "N/A"


class SessionAggregator(BaseAggregator):
    """LLM-based aggregator using the Anthropic API for narrative generation."""

    NARRATIVE_SYSTEM_PROMPT = (
        "You are a data analyst for an LLM agent observability system. "
        "You will receive aggregate statistics about agent sessions. "
        "Produce a JSON object with exactly three fields:\n"
        "- executive_summary: 2-3 paragraph natural language summary of the data\n"
        "- key_findings: list of top 5 findings as strings\n"
        "- concerns: list of safety or oversight concerns as strings\n\n"
        "Focus on autonomy patterns, oversight gaps, failure modes, and safety. "
        "Respond with ONLY valid JSON. No markdown, no explanation."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        aws_region: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for SessionAggregator. "
                "Install it with: pip install agentlens[aggregation]"
            ) from e
        if aws_region:
            self.client = anthropic.AsyncAnthropicBedrock(aws_region=aws_region)
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def aggregate(self, summaries: list[SessionSummary]) -> AggregateReport:
        stats = compute_statistics(summaries)
        narrative = await self._generate_narrative(stats, summaries)
        return AggregateReport(
            generated_at=datetime.now(timezone.utc),
            **stats,
            **narrative,
        )

    async def _generate_narrative(
        self, stats: dict, summaries: list[SessionSummary]
    ) -> dict:
        """Use LLM to generate executive_summary, key_findings, and concerns."""
        import json

        # For large batches, only include stats; for small batches include summaries
        if len(summaries) <= 50:
            summary_data = [s.model_dump() for s in summaries]
            context = json.dumps({
                "statistics": stats,
                "session_summaries": summary_data,
            }, indent=2, default=str)
        else:
            # Include stats + random sample of 20 summaries
            sample = random.sample(summaries, min(20, len(summaries)))
            summary_data = [s.model_dump() for s in sample]
            context = json.dumps({
                "statistics": stats,
                "sample_summaries": summary_data,
                "note": f"Showing 20 of {len(summaries)} sessions",
            }, indent=2, default=str)

        max_retries = 6
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        system=self.NARRATIVE_SYSTEM_PROMPT,
                        messages=[{
                            "role": "user",
                            "content": f"Generate a narrative report for this data:\n\n{context}",
                        }],
                    ),
                    timeout=120,
                )
                break
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    attempt_info = f"{attempt + 1}/{max_retries + 1}"
                    print(
                        f"API call timed out (attempt {attempt_info}), retrying...",
                        file=sys.stderr,
                    )
                    continue
                raise
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    if attempt < max_retries:
                        delay = min(2 ** attempt, 30) + random.uniform(0, 2)
                        print(
                            f"Rate limited on aggregate (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s...",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(delay)
                        continue
                raise
        raw_text = _strip_markdown_fences(response.content[0].text)
        try:
            parsed = json.loads(raw_text)
            return {
                "executive_summary": parsed["executive_summary"],
                "key_findings": parsed["key_findings"],
                "concerns": parsed["concerns"],
            }
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"Failed to parse LLM narrative response: {exc}\n"
                f"Raw LLM output:\n{response.content[0].text[:500]}"
            ) from exc
