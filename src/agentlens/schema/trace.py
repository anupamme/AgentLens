"""Core Pydantic models for the AgentLens trace schema."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.validators import is_valid_hash
from agentlens.utils.hashing import hash_content, hash_input


class ActionRecord(BaseModel):
    """A single action taken by an agent during a session."""

    action_id: str = Field(..., min_length=1, description="Unique identifier for this action")
    action_type: ActionType
    autonomy_level: AutonomyLevel
    outcome: ActionOutcome
    timestamp: datetime
    duration_ms: int = Field(..., ge=0)
    input_hash: str = Field(..., description="Hashed representation of the input")
    output_summary: str = Field(..., description="Truncated summary of the output")
    tool_name: str | None = Field(default=None, description="Name of tool used, if any")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_hash")
    @classmethod
    def validate_input_hash(cls, v: str) -> str:
        if not is_valid_hash(v):
            raise ValueError(
                "input_hash must match format '<method>:<hex>' "
                "where method is 'xxh64' or 'sha256'"
            )
        return v

    @field_validator("output_summary")
    @classmethod
    def truncate_output(cls, v: str) -> str:
        if len(v) > 500:
            return v[:497] + "..."
        return v

    @property
    def is_autonomous(self) -> bool:
        return self.autonomy_level in (
            AutonomyLevel.FULL_AUTO,
            AutonomyLevel.AUTO_WITH_AUDIT,
        )


class EscalationEvent(BaseModel):
    """An event where the agent escalated to a human."""

    timestamp: datetime
    reason: EscalationReason
    action_id: str = Field(..., description="ID of the action that triggered escalation")
    description: str = Field(..., max_length=200)
    resolved: bool = False


class SessionTrace(BaseModel):
    """Complete trace of an agent session."""

    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    agent_id: str = Field(..., min_length=1, description="Identifier for the agent")
    task_category: TaskCategory
    session_outcome: SessionOutcome
    start_time: datetime
    end_time: datetime
    actions: list[ActionRecord] = Field(..., min_length=1)
    escalations: list[EscalationEvent] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    schema_version: str = Field(default="1.0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_order(self) -> "SessionTrace":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be after start_time")
        return self

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def autonomy_ratio(self) -> float:
        if not self.actions:
            return 0.0
        autonomous = sum(1 for a in self.actions if a.is_autonomous)
        return autonomous / len(self.actions)

    @property
    def success_rate(self) -> float:
        if not self.actions:
            return 0.0
        successes = sum(1 for a in self.actions if a.outcome == ActionOutcome.SUCCESS)
        return successes / len(self.actions)

    @property
    def action_type_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for action in self.actions:
            key = action.action_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(indent=2, **kwargs)

    @classmethod
    def from_json(cls, data: str) -> "SessionTrace":
        return cls.model_validate_json(data)

    def content_hash(self) -> str:
        return hash_content(self.model_dump_json(exclude={"metadata"}))

    @classmethod
    def create_action_hash(cls, input_data: str) -> str:
        return hash_input(input_data)
