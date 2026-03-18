"""Per-Session Summarizer — produces privacy-safe SessionSummary from raw traces.

Takes a raw SessionTrace and produces a structured SessionSummary using an LLM.
The SessionSummary is the ONLY human-readable representation of a session.
Raw traces are never shown to humans.

Usage:
    summarizer = SessionSummarizer(api_key="...", model="claude-haiku-4-5-20251001")
    summary = await summarizer.summarize(trace)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import Counter

import re

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
)
from agentlens.schema.trace import SessionTrace
from agentlens.aggregation.models import (
    AUTONOMY_KEY_MAP,
    CONSEQUENTIAL_ACTION_TYPES,
    SessionSummary,
)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _compute_base_fields(trace: SessionTrace) -> dict:
    """Compute all deterministic summary fields from a SessionTrace."""
    actions = trace.actions

    # Autonomy distribution
    autonomy_counts: dict[str, int] = {v: 0 for v in AUTONOMY_KEY_MAP.values()}
    for action in actions:
        key = AUTONOMY_KEY_MAP[action.autonomy_level]
        autonomy_counts[key] += 1
    total = len(actions)
    autonomy_distribution = {k: v / total for k, v in autonomy_counts.items()}

    # Tool usage
    tool_actions = [a for a in actions if a.tool_name is not None]
    tools_used = sorted(set(a.tool_name for a in tool_actions))
    tool_call_count = len(tool_actions)
    tool_successes = sum(1 for a in tool_actions if a.outcome == ActionOutcome.SUCCESS)
    tool_success_rate = tool_successes / tool_call_count if tool_call_count > 0 else 1.0

    # Failures
    failed_actions = [a for a in actions if a.outcome == ActionOutcome.FAILURE]
    failure_count = len(failed_actions)
    failure_types = sorted(set(
        a.metadata.get("error_type", "unknown") for a in failed_actions
    ))

    # Escalations
    escalation_count = len(trace.escalations)
    escalation_reasons = sorted(set(e.reason.value for e in trace.escalations))

    # Did it fail gracefully? True if every failure had a corresponding escalation,
    # or if there were no failures.
    failed_action_ids = {a.action_id for a in failed_actions}
    escalated_action_ids = {e.action_id for e in trace.escalations}
    did_fail_gracefully = (
        failure_count == 0 or failed_action_ids.issubset(escalated_action_ids)
    )

    # Performance
    duration_seconds = trace.duration_ms / 1000.0
    total_latency_ms = sum(a.duration_ms for a in actions)

    # Consequential actions and oversight gap
    consequential_actions = [
        a for a in actions if a.action_type in CONSEQUENTIAL_ACTION_TYPES
    ]
    consequential_action_count = len(consequential_actions)
    unsupervised_consequential_count = sum(
        1 for a in consequential_actions
        if a.autonomy_level == AutonomyLevel.FULL_AUTO
    )
    oversight_gap_score = (
        unsupervised_consequential_count / max(consequential_action_count, 1)
    )

    # Action sequence summary (arrow-separated flow of action types)
    action_sequence_summary = " → ".join(
        a.action_type.value.capitalize() for a in actions
    )

    return {
        "session_id": trace.session_id,
        "agent_type": trace.agent_id,
        "task_category": trace.task_category,
        "start_time": trace.start_time,
        "end_time": trace.end_time,
        "total_actions": total,
        "autonomy_distribution": autonomy_distribution,
        "tools_used": tools_used,
        "tool_call_count": tool_call_count,
        "tool_success_rate": round(tool_success_rate, 4),
        "failure_count": failure_count,
        "failure_types": failure_types,
        "escalation_count": escalation_count,
        "escalation_reasons": escalation_reasons,
        "did_fail_gracefully": did_fail_gracefully,
        "duration_seconds": round(duration_seconds, 2),
        "total_latency_ms": total_latency_ms,
        "session_outcome": trace.session_outcome,
        "consequential_action_count": consequential_action_count,
        "unsupervised_consequential_count": unsupervised_consequential_count,
        "oversight_gap_score": round(oversight_gap_score, 4),
        "action_sequence_summary": action_sequence_summary,
    }


class BaseSummarizer(ABC):
    """Abstract base for session summarizers."""

    @abstractmethod
    async def summarize(self, trace: SessionTrace) -> SessionSummary:
        ...

    async def summarize_batch(
        self, traces: list[SessionTrace], max_concurrent: int = 5
    ) -> list[SessionSummary]:
        """Summarize multiple traces with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _summarize_one(trace: SessionTrace) -> SessionSummary:
            async with semaphore:
                return await self.summarize(trace)

        return await asyncio.gather(*[_summarize_one(t) for t in traces])


class MockSummarizer(BaseSummarizer):
    """Deterministic summarizer for testing. No LLM calls."""

    async def summarize(self, trace: SessionTrace) -> SessionSummary:
        fields = _compute_base_fields(trace)

        # Deterministic abstract — no PII, just generic description
        category = trace.task_category.value.replace("_", " ")
        fields["task_abstract"] = (
            f"Agent performed {fields['total_actions']} actions "
            f"on a {category} task"
        )

        return SessionSummary(**fields)


class SessionSummarizer(BaseSummarizer):
    """LLM-based summarizer using the Anthropic API."""

    SYSTEM_PROMPT = (
        "You are a privacy-preserving trace analyzer for an LLM agent observability system.\n\n"
        "You will receive a structured JSON trace of an LLM agent session. Your job is to produce "
        "a structured summary that:\n"
        "1. NEVER includes any raw user input, personal information, code content, file names, "
        "or identifiable details\n"
        "2. Abstracts actions into generic descriptions (e.g., \"reviewed a Python file\" not "
        "\"reviewed auth_controller.py\")\n"
        "3. Accurately computes autonomy distributions, failure rates, and oversight metrics\n"
        "4. Assesses which actions were \"consequential\" (modified external state: wrote files, "
        "posted comments, sent messages, made API calls that change data) vs \"read-only\" "
        "(searched, read, analyzed)\n"
        "5. Computes the oversight gap score: "
        "(unsupervised_consequential_count / max(consequential_action_count, 1))\n\n"
        "Respond with ONLY a JSON object matching the SessionSummary schema. "
        "No markdown, no explanation."
    )

    USER_PROMPT_TEMPLATE = (
        "Analyze this agent session trace and produce a SessionSummary.\n\n"
        "Trace:\n{trace_json}\n\n"
        "Respond with ONLY a valid JSON object. Fields required:\n"
        "- session_id, agent_type, task_category\n"
        "- task_abstract (generic description, NO PII or specific identifiers)\n"
        "- action_sequence_summary (arrow-separated flow)\n"
        "- total_actions, autonomy_distribution (as fractions summing to 1.0)\n"
        "- tools_used, tool_call_count, tool_success_rate\n"
        "- failure_count, failure_types, escalation_count, escalation_reasons, did_fail_gracefully\n"
        "- duration_seconds, total_latency_ms\n"
        "- session_outcome\n"
        "- consequential_action_count, unsupervised_consequential_count, oversight_gap_score"
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
                "The 'anthropic' package is required for SessionSummarizer. "
                "Install it with: pip install agentlens[aggregation]"
            ) from e
        if aws_region:
            self.client = anthropic.AsyncAnthropicBedrock(aws_region=aws_region)
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def summarize(self, trace: SessionTrace) -> SessionSummary:
        """Summarize a single session trace into a privacy-safe summary."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": self.USER_PROMPT_TEMPLATE.format(
                    trace_json=trace.to_json()
                ),
            }],
        )
        raw_text = _strip_markdown_fences(response.content[0].text)
        try:
            return SessionSummary.model_validate_json(raw_text)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse LLM summary for session {trace.session_id}: {exc}\n"
                f"Raw LLM output:\n{response.content[0].text[:500]}"
            ) from exc
