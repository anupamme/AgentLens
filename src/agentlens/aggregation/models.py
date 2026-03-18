"""Pydantic models for the aggregation pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from agentlens.schema.enums import (
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)

# Actions that modify external state (files, APIs, messages, etc.)
CONSEQUENTIAL_ACTION_TYPES = frozenset({
    ActionType.WRITE,
    ActionType.EXECUTE,
    ActionType.COMMUNICATE,
})

# Map AutonomyLevel enum values to human-readable keys for distributions
AUTONOMY_KEY_MAP: dict[AutonomyLevel, str] = {
    AutonomyLevel.FULL_AUTO: "fully_autonomous",
    AutonomyLevel.AUTO_WITH_AUDIT: "auto_with_audit",
    AutonomyLevel.HUMAN_CONFIRMED: "human_confirmed",
    AutonomyLevel.HUMAN_DRIVEN: "human_driven",
}


class SessionSummary(BaseModel):
    """Privacy-safe summary of a single agent session. This is what humans see."""

    session_id: str
    agent_type: str
    task_category: TaskCategory

    # --- What happened (abstract, no PII) ---
    task_abstract: str
    action_sequence_summary: str

    # --- Autonomy Profile ---
    total_actions: int
    autonomy_distribution: dict[str, float]

    # --- Tool Usage ---
    tools_used: list[str]
    tool_call_count: int
    tool_success_rate: float

    # --- Failures and Escalations ---
    failure_count: int
    failure_types: list[str]
    escalation_count: int
    escalation_reasons: list[str]
    did_fail_gracefully: bool

    # --- Performance ---
    duration_seconds: float
    total_latency_ms: int

    # --- Outcome ---
    session_outcome: SessionOutcome

    # --- Safety Assessment ---
    consequential_action_count: int
    unsupervised_consequential_count: int
    oversight_gap_score: float

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> SessionSummary:
        return cls.model_validate_json(data)


class AggregateReport(BaseModel):
    """Aggregate statistics across multiple sessions. The primary output for researchers."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime
    session_count: int
    time_range: dict

    # --- Task Distribution ---
    task_category_distribution: dict[str, int]
    agent_type_distribution: dict[str, int]

    # --- Autonomy Overview ---
    mean_autonomous_action_ratio: float
    autonomy_histogram: dict[str, float]

    # --- Tool Usage Patterns ---
    tool_usage_ranking: list[dict]
    most_common_tool_sequences: list[str]

    # --- Failure Taxonomy ---
    failure_rate_by_agent: dict[str, float]
    failure_type_distribution: dict[str, int]
    graceful_failure_rate: float

    # --- Escalation Analysis ---
    mean_escalation_rate: float
    escalation_reason_distribution: dict[str, int]

    # --- Oversight Gap ---
    mean_oversight_gap_score: float
    oversight_gap_by_agent: dict[str, float]
    oversight_gap_by_task_category: dict[str, float]
    high_risk_sessions: int

    # --- Performance ---
    mean_duration_seconds: float
    mean_actions_per_session: float
    outcome_distribution: dict[str, int]

    # --- Narrative Summary (LLM-generated) ---
    executive_summary: str
    key_findings: list[str]
    concerns: list[str]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> AggregateReport:
        return cls.model_validate_json(data)
