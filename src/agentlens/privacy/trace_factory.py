"""Shared helper to generate diverse synthetic traces for privacy testing."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, EscalationEvent, SessionTrace
from agentlens.utils.hashing import hash_input


def make_diverse_traces(count: int = 50, seed: int = 42) -> list[SessionTrace]:
    """Generate varied traces with different categories, action types, tools, etc."""
    rng = random.Random(seed)

    categories = list(TaskCategory)
    outcomes = [SessionOutcome.SUCCESS, SessionOutcome.SUCCESS, SessionOutcome.PARTIAL,
                SessionOutcome.FAILURE, SessionOutcome.ESCALATED]
    action_types = list(ActionType)
    autonomy_levels = list(AutonomyLevel)
    action_outcomes = [ActionOutcome.SUCCESS, ActionOutcome.SUCCESS, ActionOutcome.SUCCESS,
                       ActionOutcome.FAILURE, ActionOutcome.PARTIAL]
    tool_names = ["search_tool", "write_tool", "bash", "git", "browser", "editor",
                  "code_analyzer", "test_runner", None, None]
    agent_ids = ["code-reviewer", "researcher", "sys-admin", "data-analyst", "communicator"]
    escalation_reasons = list(EscalationReason)

    traces: list[SessionTrace] = []
    base_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(count):
        num_actions = rng.randint(1, 8)
        start_time = base_time + timedelta(hours=i)
        category = rng.choice(categories)
        agent_id = rng.choice(agent_ids)

        actions: list[ActionRecord] = []
        current_time = start_time

        for j in range(num_actions):
            duration = rng.randint(50, 5000)
            current_time = current_time + timedelta(milliseconds=duration)
            tool = rng.choice(tool_names)
            outcome = rng.choice(action_outcomes)
            metadata = {}
            if outcome == ActionOutcome.FAILURE:
                metadata["error_type"] = rng.choice(["timeout", "api_error", "parse_error"])

            actions.append(ActionRecord(
                action_id=f"act-{i:03d}-{j:03d}",
                action_type=rng.choice(action_types),
                autonomy_level=rng.choice(autonomy_levels),
                outcome=outcome,
                timestamp=current_time,
                duration_ms=duration,
                input_hash=hash_input(f"input-{i}-{j}-{rng.randint(0, 10000)}"),
                output_summary=f"Action {j} output for {category.value} task",
                tool_name=tool,
                metadata=metadata,
            ))

        end_time = current_time + timedelta(seconds=rng.randint(1, 60))

        # Add escalations for some traces
        escalations: list[EscalationEvent] = []
        if rng.random() < 0.3:
            failed_ids = [a.action_id for a in actions if a.outcome == ActionOutcome.FAILURE]
            esc_id = rng.choice(failed_ids) if failed_ids else actions[0].action_id
            escalations.append(EscalationEvent(
                timestamp=current_time,
                reason=rng.choice(escalation_reasons),
                action_id=esc_id,
                description=f"Escalation for session {i}",
            ))

        session_outcome = rng.choice(outcomes)

        traces.append(SessionTrace(
            session_id=f"sess-{i:04d}",
            agent_id=agent_id,
            task_category=category,
            session_outcome=session_outcome,
            start_time=start_time,
            end_time=end_time,
            actions=actions,
            escalations=escalations,
            total_tokens=rng.randint(100, 50000),
        ))

    return traces
