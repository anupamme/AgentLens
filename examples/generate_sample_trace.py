#!/usr/bin/env python3
"""Generate and print a sample SessionTrace."""

from datetime import datetime, timezone

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, SessionTrace
from agentlens.utils.hashing import hash_input


def main() -> None:
    actions = [
        ActionRecord(
            action_id="act-001",
            action_type=ActionType.READ,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            timestamp=datetime(2025, 3, 1, 9, 0, 0, tzinfo=timezone.utc),
            duration_ms=150,
            input_hash=hash_input("Read the project requirements document"),
            output_summary="Read requirements doc: 12 user stories, 3 milestones",
        ),
        ActionRecord(
            action_id="act-002",
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            timestamp=datetime(2025, 3, 1, 9, 1, 0, tzinfo=timezone.utc),
            duration_ms=2000,
            input_hash=hash_input("Analyze requirements for implementation plan"),
            output_summary="Identified 5 key components and their dependencies",
        ),
        ActionRecord(
            action_id="act-003",
            action_type=ActionType.WRITE,
            autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
            outcome=ActionOutcome.SUCCESS,
            timestamp=datetime(2025, 3, 1, 9, 3, 0, tzinfo=timezone.utc),
            duration_ms=3500,
            input_hash=hash_input("Generate implementation plan document"),
            output_summary="Wrote implementation plan with timeline and task breakdown",
        ),
    ]

    session = SessionTrace(
        session_id="sess-sample-001",
        agent_id="planning-agent-v1",
        task_category=TaskCategory.CODE_GENERATION,
        session_outcome=SessionOutcome.SUCCESS,
        start_time=datetime(2025, 3, 1, 9, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 3, 1, 9, 5, 0, tzinfo=timezone.utc),
        actions=actions,
        total_tokens=2500,
    )

    print("=== Sample SessionTrace ===\n")
    print(f"Session ID:  {session.session_id}")
    print(f"Agent:       {session.agent_id}")
    print(f"Category:    {session.task_category.value}")
    print(f"Outcome:     {session.session_outcome.value}")
    print(f"Duration:    {session.duration_ms}ms")
    print(f"Actions:     {len(session.actions)}")
    print(f"Autonomy:    {session.autonomy_ratio:.0%}")
    print(f"Success:     {session.success_rate:.0%}")
    print(f"Distribution: {session.action_type_distribution}")
    print(f"Content hash: {session.content_hash()}")
    print(f"\n=== JSON ===\n")
    print(session.to_json())


if __name__ == "__main__":
    main()
