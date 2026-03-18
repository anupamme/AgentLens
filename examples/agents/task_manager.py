"""Task Manager Agent — manages an in-memory task store with scripted commands.

Uses AnthropicBedrock client with AgentLens tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import ANTHROPIC_MODEL_ID, TRACES_DIR

from agentlens.schema.enums import (
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.sdk.tracer import AgentTracer

SCRIPTED_COMMANDS = [
    {"type": "add", "title": "Write unit tests", "priority": 1, "deadline": "2026-03-19"},
    {"type": "add", "title": "Deploy to staging", "priority": 2, "deadline": "2026-03-19"},
    {"type": "add", "title": "Update docs", "priority": 3, "deadline": "2026-03-20"},
    {"type": "reorder", "reason": "Deploy should come after tests"},
    {"type": "add", "title": "Fix login bug", "priority": 1, "deadline": "2026-03-19"},
    {"type": "complete_overdue"},
]


def run() -> None:
    tracer = AgentTracer(
        agent_type="task_manager",
        task_category=TaskCategory.OTHER,
        model_used=ANTHROPIC_MODEL_ID,
    )
    tracer.start_session("Manage task list: process user commands for project tasks")

    tasks: list[dict] = []

    for cmd in SCRIPTED_COMMANDS:
        if cmd["type"] == "add":
            with tracer.action(
                action_type=ActionType.WRITE,
                autonomy_level=AutonomyLevel.HUMAN_DRIVEN,
                raw_input=f"Add task: {cmd['title']} (P{cmd['priority']}, due {cmd['deadline']})",
                tool_name="add_task",
            ) as ctx:
                task = {
                    "id": len(tasks) + 1,
                    "title": cmd["title"],
                    "priority": cmd["priority"],
                    "deadline": cmd["deadline"],
                    "done": False,
                }
                tasks.append(task)
                ctx.set_output_summary(f"Added task #{task['id']}: {task['title']}")

        elif cmd["type"] == "reorder":
            # Agent suggests reordering — requires human confirmation
            with tracer.action(
                action_type=ActionType.REASON,
                autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
                raw_input=f"Suggest reorder: {cmd['reason']}",
                tool_name="suggest_reorder",
            ) as ctx:
                # Simulate: swap priorities of task 1 and 2
                if len(tasks) >= 2:
                    tasks[0]["priority"], tasks[1]["priority"] = (
                        tasks[1]["priority"],
                        tasks[0]["priority"],
                    )
                ctx.set_output_summary(
                    "Reordered: 'Write unit tests' now before 'Deploy to staging'"
                )

        elif cmd["type"] == "complete_overdue":
            # Auto-complete overdue tasks (FULL_AUTO)
            completed = 0
            with tracer.action(
                action_type=ActionType.EXECUTE,
                autonomy_level=AutonomyLevel.FULL_AUTO,
                raw_input="Auto-complete overdue tasks before 2026-03-19",
                tool_name="auto_complete_overdue",
            ) as ctx:
                for task in tasks:
                    if task["deadline"] <= "2026-03-19" and not task["done"]:
                        task["done"] = True
                        completed += 1
                ctx.set_output_summary(f"Auto-completed {completed} overdue tasks")

    # Detect conflicting deadlines → escalate
    deadline_counts: dict[str, int] = {}
    for t in tasks:
        deadline_counts[t["deadline"]] = deadline_counts.get(t["deadline"], 0) + 1
    conflicts = {d: c for d, c in deadline_counts.items() if c > 2}

    if conflicts:
        tracer.record_escalation(
            reason=EscalationReason.POLICY_REQUIRED,
            context_summary=(
                f"Conflicting deadlines: "
                f"{next(iter(conflicts.values()))}+ tasks due same date"
            ),
        )

    trace = tracer.end_session(SessionOutcome.SUCCESS, user_satisfaction_proxy=0.9)

    output_dir = str(TRACES_DIR / "task_manager")
    tracer.save_json(output_dir)
    print(f"Trace saved to {output_dir}/{trace.session_id}.json")
    print(f"Actions: {len(trace.actions)}, Escalations: {len(trace.escalations)}")
    print(f"Tasks: {[t['title'] + (' [done]' if t['done'] else '') for t in tasks]}")


if __name__ == "__main__":
    run()
