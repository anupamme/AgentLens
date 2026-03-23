"""Pydantic result models for the five-dimensional agent oversight analysis."""

from __future__ import annotations

import math
import statistics
from typing import Any

from pydantic import BaseModel, Field


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation using stdlib. Returns 0.0 for degenerate cases."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


class AutonomyAnalysis(BaseModel):
    """Dimension 1: Autonomy Profiling results."""

    overall_distribution: dict[str, float] = Field(default_factory=dict)
    by_agent: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_task_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    autonomy_ratio_histogram: list[float] = Field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    high_autonomy_session_count: int = 0
    high_autonomy_fraction: float = 0.0
    high_autonomy_by_agent: dict[str, int] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> AutonomyAnalysis:
        return cls.model_validate_json(data)


class FailureAnalysis(BaseModel):
    """Dimension 2: Failure Taxonomy results."""

    overall_failure_rate: float = 0.0
    failure_rate_by_agent: dict[str, float] = Field(default_factory=dict)
    failure_rate_by_task: dict[str, float] = Field(default_factory=dict)
    failure_type_counts: dict[str, int] = Field(default_factory=dict)
    failure_type_by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)
    graceful_failure_rate: float = 0.0
    silent_failure_rate: float = 0.0
    graceful_by_agent: dict[str, float] = Field(default_factory=dict)
    failure_rate_by_autonomy_level: dict[str, float] = Field(default_factory=dict)
    autonomous_vs_supervised_failure_ratio: float = 0.0

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> FailureAnalysis:
        return cls.model_validate_json(data)


class ToolUsageAnalysis(BaseModel):
    """Dimension 3: Tool Usage Patterns results."""

    tool_frequency: dict[str, int] = Field(default_factory=dict)
    tool_frequency_by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)
    tool_success_rates: dict[str, float] = Field(default_factory=dict)
    problematic_tools: list[str] = Field(default_factory=list)
    common_bigrams: list[tuple[str, int]] = Field(default_factory=list)
    common_trigrams: list[tuple[str, int]] = Field(default_factory=list)
    avg_unique_tools_per_session: float = 0.0
    tool_to_economic_task: dict[str, str] = Field(default_factory=dict)
    economic_task_distribution: dict[str, int] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> ToolUsageAnalysis:
        return cls.model_validate_json(data)


class EscalationAnalysis(BaseModel):
    """Dimension 4: Escalation Analysis results."""

    overall_escalation_rate: float = 0.0
    escalation_rate_by_agent: dict[str, float] = Field(default_factory=dict)
    reason_distribution: dict[str, int] = Field(default_factory=dict)
    reason_by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)
    false_escalation_estimate: float = 0.0
    missed_escalation_estimate: float = 0.0
    missed_escalation_by_agent: dict[str, float] = Field(default_factory=dict)
    mean_actions_before_first_escalation: float = 0.0

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> EscalationAnalysis:
        return cls.model_validate_json(data)


class OversightGapAnalysis(BaseModel):
    """Dimension 5: Oversight Gap Score results."""

    mean_score: float = 0.0
    median_score: float = 0.0
    std_score: float = 0.0
    score_histogram: list[float] = Field(default_factory=list)
    by_agent: dict[str, float] = Field(default_factory=dict)
    by_task_category: dict[str, float] = Field(default_factory=dict)
    by_agent_by_task: dict[str, dict[str, float]] = Field(default_factory=dict)
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0
    risk_tier_by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)
    gap_vs_failure: float = 0.0
    gap_vs_duration: float = 0.0
    gap_vs_action_count: float = 0.0
    top_risk_sessions: list[dict[str, Any]] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> OversightGapAnalysis:
        return cls.model_validate_json(data)


class AnalysisResults(BaseModel):
    """Combined results from all five dimensions of analysis."""

    autonomy: AutonomyAnalysis = Field(default_factory=AutonomyAnalysis)
    failures: FailureAnalysis = Field(default_factory=FailureAnalysis)
    tools: ToolUsageAnalysis = Field(default_factory=ToolUsageAnalysis)
    escalations: EscalationAnalysis = Field(default_factory=EscalationAnalysis)
    oversight_gap: OversightGapAnalysis = Field(default_factory=OversightGapAnalysis)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> AnalysisResults:
        return cls.model_validate_json(data)
