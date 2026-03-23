"""Utility vs privacy trade-off analysis across granularity levels."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentlens.aggregation.models import SessionSummary
from agentlens.aggregation.summarizer import BaseSummarizer, _compute_base_fields
from agentlens.privacy.leakage_test import check_summary_for_pii
from agentlens.privacy.pii_generator import PIIGenerator
from agentlens.privacy.trace_factory import make_diverse_traces
from agentlens.schema.trace import SessionTrace


@dataclass
class UtilityPrivacyReport:
    """Results from utility-privacy trade-off analysis."""

    levels: list[int] = field(default_factory=list)
    mean_utility_scores: list[float] = field(default_factory=list)
    mean_leakage_rates: list[float] = field(default_factory=list)
    utility_std: list[float] = field(default_factory=list)
    leakage_std: list[float] = field(default_factory=list)
    recommended_level: int = 3
    recommendation_rationale: str = ""


# Level-specific task abstract templates
_LEVEL_ABSTRACTS = {
    1: "An agent performed a task and completed it.",
    2: "A {category} agent analyzed files and found issues.",
    3: "Agent performed {num_actions} actions on a {category} task",
    4: "Agent performed {num_actions} actions using {tools} on a {category} task",
    5: (
        "Agent performed {num_actions} actions using {tools} on a {category} task."
        " Output: {output_snippet}"
    ),
}


class GranularityMockSummarizer(BaseSummarizer):
    """Mock summarizer that varies detail level by granularity setting."""

    def __init__(self, level: int = 3) -> None:
        if level < 1 or level > 5:
            raise ValueError("Granularity level must be between 1 and 5")
        self.level = level

    async def summarize(self, trace: SessionTrace) -> SessionSummary:
        fields = _compute_base_fields(trace)

        category = trace.task_category.value.replace("_", " ")
        num_actions = fields["total_actions"]
        tools = ", ".join(fields["tools_used"]) if fields["tools_used"] else "no tools"

        if self.level == 1:
            abstract = _LEVEL_ABSTRACTS[1]
        elif self.level == 2:
            abstract = _LEVEL_ABSTRACTS[2].format(category=category)
        elif self.level == 3:
            abstract = _LEVEL_ABSTRACTS[3].format(
                num_actions=num_actions, category=category,
            )
        elif self.level == 4:
            abstract = _LEVEL_ABSTRACTS[4].format(
                num_actions=num_actions, tools=tools, category=category,
            )
        else:  # level 5
            # Include output_summary snippets (may contain injected PII)
            snippets = [a.output_summary[:50] for a in trace.actions[:3]]
            output_snippet = "; ".join(snippets)
            abstract = _LEVEL_ABSTRACTS[5].format(
                num_actions=num_actions, tools=tools,
                category=category, output_snippet=output_snippet,
            )

        fields["task_abstract"] = abstract
        return SessionSummary(**fields)


def _rate_utility_heuristic(summary: SessionSummary, level: int) -> float:
    """Rate utility of a summary on a 0-5 scale using heuristics."""
    score = 0.0

    # Base score by level
    score += level * 0.6

    # Bonus for having tool information
    if summary.tools_used:
        score += 0.3

    # Bonus for non-trivial abstract
    if len(summary.task_abstract) > 30:
        score += 0.3

    # Bonus for action sequence information
    if len(summary.action_sequence_summary) > 20:
        score += 0.4

    return min(score, 5.0)


class UtilityPrivacyAnalysis:
    """Analyzes the trade-off between utility and privacy across granularity levels."""

    def __init__(self, pii_generator: PIIGenerator) -> None:
        self.pii_generator = pii_generator

    async def run(
        self,
        traces: list[SessionTrace] | None = None,
        granularity_levels: list[int] | None = None,
        num_traces: int = 30,
    ) -> UtilityPrivacyReport:
        """Run utility-privacy analysis across granularity levels."""
        if granularity_levels is None:
            granularity_levels = [1, 2, 3, 4, 5]

        if traces is None:
            traces = make_diverse_traces(count=num_traces, seed=99)

        # Generate PII and inject
        pii_bundles = self.pii_generator.generate(count=len(traces))
        injected_traces: list[SessionTrace] = []
        all_pii_per_trace: list[set[str]] = []

        for trace, pii in zip(traces, pii_bundles):
            injected = self.pii_generator.inject_into_trace(trace, pii)
            injected_traces.append(injected)
            all_pii_per_trace.append(PIIGenerator.get_all_pii_strings(pii))

        report = UtilityPrivacyReport(levels=granularity_levels)

        best_level = granularity_levels[0]
        best_utility = 0.0

        for level in granularity_levels:
            summarizer = GranularityMockSummarizer(level=level)
            summaries = await summarizer.summarize_batch(injected_traces)

            utilities: list[float] = []
            leakages: list[float] = []

            for summary, pii_strings in zip(summaries, all_pii_per_trace):
                utility = _rate_utility_heuristic(summary, level)
                utilities.append(utility)

                detected = check_summary_for_pii(summary, pii_strings)
                leakages.append(1.0 if detected else 0.0)

            mean_utility = sum(utilities) / len(utilities)
            mean_leakage = sum(leakages) / len(leakages)

            # Standard deviation
            utility_std = _std(utilities)
            leakage_std = _std(leakages)

            report.mean_utility_scores.append(round(mean_utility, 4))
            report.mean_leakage_rates.append(round(mean_leakage, 4))
            report.utility_std.append(round(utility_std, 4))
            report.leakage_std.append(round(leakage_std, 4))

            # Track best level: highest utility where leakage is near-zero
            if mean_leakage < 0.01 and mean_utility > best_utility:
                best_utility = mean_utility
                best_level = level

        report.recommended_level = best_level
        report.recommendation_rationale = (
            f"Level {best_level} provides the best utility "
            f"(score={best_utility:.2f}) with near-zero PII leakage."
        )

        return report


def _std(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
