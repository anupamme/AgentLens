"""Enumerations for the AgentLens trace schema."""

from enum import Enum


class ActionType(str, Enum):
    """Types of actions an agent can take."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEARCH = "search"
    COMMUNICATE = "communicate"
    REASON = "reason"


class AutonomyLevel(str, Enum):
    """Level of human oversight for an action."""

    FULL_AUTO = "full_auto"
    AUTO_WITH_AUDIT = "auto_with_audit"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_DRIVEN = "human_driven"


class ActionOutcome(str, Enum):
    """Outcome of an individual action."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class SessionOutcome(str, Enum):
    """Overall outcome of an agent session."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class TaskCategory(str, Enum):
    """High-level category of the agent's task."""

    CODE_REVIEW = "code_review"
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    SYSTEM_ADMIN = "system_admin"
    OTHER = "other"


class EscalationReason(str, Enum):
    """Reason for escalating to a human."""

    CONFIDENCE_LOW = "confidence_low"
    RISK_HIGH = "risk_high"
    POLICY_REQUIRED = "policy_required"
    USER_REQUESTED = "user_requested"
    ERROR_REPEATED = "error_repeated"
    SCOPE_EXCEEDED = "scope_exceeded"
