"""Trace validator — validates batches of SessionTrace objects."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from agentlens.schema.enums import ActionOutcome, SessionOutcome
from agentlens.schema.trace import SessionTrace

MAX_ACTIONS_PER_SESSION = 100
MAX_DURATION_SECONDS = 600


class ValidationReport(BaseModel):
    """Report summarizing validation results for a batch of traces."""

    total_traces: int = 0
    valid_traces: int = 0
    validation_rate: float = 0.0
    errors_by_type: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    diversity_score: float = 0.0
    agent_distribution: dict[str, int] = Field(default_factory=dict)


class TraceValidator:
    """Validates batches of SessionTrace objects for quality and consistency."""

    def validate_batch(self, traces: list[SessionTrace]) -> ValidationReport:
        """Validate a batch of traces and return a report."""
        errors: dict[str, int] = Counter()
        warnings: list[str] = []
        valid_count = 0
        session_ids: set[str] = set()
        categories: set[str] = set()
        agent_counts: dict[str, int] = Counter()

        for trace in traces:
            trace_errors = self._validate_single(trace, session_ids)
            if trace_errors:
                for err in trace_errors:
                    errors[err] += 1
            else:
                valid_count += 1

            session_ids.add(trace.session_id)
            categories.add(trace.task_category.value)
            agent_counts[trace.agent_id] += 1

        # Diversity check
        num_categories = len(categories)
        if num_categories < 3:
            warnings.append(
                f"Low diversity: only {num_categories} distinct task categories "
                f"(recommend >= 3)"
            )

        # Diversity score: normalized by number of TaskCategory values (7)
        diversity_score = min(num_categories / 3.0, 1.0)

        total = len(traces)
        return ValidationReport(
            total_traces=total,
            valid_traces=valid_count,
            validation_rate=round(valid_count / max(total, 1), 4),
            errors_by_type=dict(errors),
            warnings=warnings,
            diversity_score=round(diversity_score, 4),
            agent_distribution=dict(agent_counts),
        )

    def _validate_single(
        self,
        trace: SessionTrace,
        seen_ids: set[str],
    ) -> list[str]:
        """Validate a single trace. Returns list of error types (empty = valid)."""
        errors: list[str] = []

        # 1. Deduplication
        if trace.session_id in seen_ids:
            errors.append("duplicate_session_id")

        # 2. Temporal consistency — action timestamps monotonically increasing
        for i in range(1, len(trace.actions)):
            if trace.actions[i].timestamp < trace.actions[i - 1].timestamp:
                errors.append("timestamps_not_monotonic")
                break

        # 3. Duration sanity
        duration_s = trace.duration_ms / 1000.0
        if duration_s <= 0:
            errors.append("duration_zero_or_negative")
        if duration_s > MAX_DURATION_SECONDS:
            errors.append("duration_exceeds_max")

        # 4. Action count sanity
        num_actions = len(trace.actions)
        if num_actions < 1:
            errors.append("no_actions")
        if num_actions > MAX_ACTIONS_PER_SESSION:
            errors.append("too_many_actions")

        # 5. Autonomy annotation completeness
        for action in trace.actions:
            if action.autonomy_level is None:
                errors.append("missing_autonomy_level")
                break

        # 6. Outcome consistency
        all_success = all(
            a.outcome == ActionOutcome.SUCCESS for a in trace.actions
        )
        if all_success and trace.session_outcome == SessionOutcome.FAILURE:
            errors.append("outcome_inconsistency")

        return errors
