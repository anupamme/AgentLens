"""Adversarial re-identification attack experiment."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agentlens.aggregation.aggregator import BaseAggregator
from agentlens.aggregation.models import AggregateReport, SessionSummary
from agentlens.aggregation.summarizer import BaseSummarizer
from agentlens.schema.trace import SessionTrace


@dataclass
class ReidentificationResult:
    """Results from a single re-identification attack."""

    batch_size: int = 0
    num_targets: int = 0
    num_decoys: int = 0
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    adversary_confidence_mean: float = 0.0
    random_baseline_tpr: float = 0.0


class BaseAdversary(ABC):
    """Abstract base for re-identification adversaries."""

    @abstractmethod
    async def attack(
        self,
        candidate_summaries: list[SessionSummary],
        report: AggregateReport,
        num_targets: int,
    ) -> list[dict[str, Any]]:
        """Return list of dicts with 'session_id' and 'confidence' for guessed targets."""
        ...


class MockAdversary(BaseAdversary):
    """Random-guessing adversary for deterministic testing."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    async def attack(
        self,
        candidate_summaries: list[SessionSummary],
        report: AggregateReport,
        num_targets: int,
    ) -> list[dict[str, Any]]:
        all_ids = [s.session_id for s in candidate_summaries]
        # Guess at the base rate: num_targets / total
        guess_rate = num_targets / len(all_ids) if all_ids else 0.5
        guesses: list[dict[str, Any]] = []
        for sid in all_ids:
            if self._rng.random() < guess_rate:
                guesses.append({
                    "session_id": sid,
                    "confidence": round(self._rng.uniform(0.1, 0.9), 3),
                })
        return guesses


class LLMAdversary(BaseAdversary):
    """LLM-powered adversary that tries to identify target sessions."""

    # Key fields to include in the compact candidate representation
    _CANDIDATE_FIELDS = (
        "session_id", "agent_type", "task_category", "task_abstract",
        "total_actions", "tools_used", "session_outcome",
    )

    # Key fields to include in the compact report representation
    _REPORT_FIELDS = (
        "session_count", "task_category_distribution", "agent_type_distribution",
        "tool_usage_ranking", "outcome_distribution", "executive_summary",
        "key_findings", "concerns",
    )

    MAX_CANDIDATES = 50

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for LLMAdversary. "
                "Install it with: pip install agentlens[privacy]"
            ) from e
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def _compact_candidate(self, summary: SessionSummary) -> dict[str, object]:
        """Extract only the fields relevant for re-identification."""
        full = summary.model_dump()
        return {k: full[k] for k in self._CANDIDATE_FIELDS if k in full}

    def _compact_report(self, report: AggregateReport) -> dict[str, object]:
        """Extract only the fields relevant for re-identification."""
        full = report.model_dump()
        return {k: full[k] for k in self._REPORT_FIELDS if k in full}

    async def attack(
        self,
        candidate_summaries: list[SessionSummary],
        report: AggregateReport,
        num_targets: int,
    ) -> list[dict[str, Any]]:
        import json

        # Sample candidates if the list is too large
        candidates = candidate_summaries
        all_ids = [s.session_id for s in candidate_summaries]
        if len(candidates) > self.MAX_CANDIDATES:
            rng = random.Random(0)
            candidates = rng.sample(candidate_summaries, self.MAX_CANDIDATES)

        candidates_json = json.dumps(
            [self._compact_candidate(s) for s in candidates],
            indent=2, default=str,
        )
        report_json = json.dumps(self._compact_report(report), indent=2, default=str)

        prompt = (
            f"You are a privacy auditor testing whether an aggregate report leaks information "
            f"about individual sessions.\n\n"
            f"Below is an aggregate report generated from {num_targets} target sessions "
            f"(mixed with decoy sessions). Your goal is to identify which of the candidate "
            f"sessions were part of the target batch that generated the report.\n\n"
            f"Aggregate Report:\n{report_json}\n\n"
            f"Candidate Sessions ({len(candidates)} of {len(all_ids)} total):\n"
            f"{candidates_json}\n\n"
            f"Respond with ONLY a JSON array of objects, each with 'session_id' (string) "
            f"and 'confidence' (float 0-1) for sessions you believe were targets. "
            f"Only include sessions where confidence > 0.5."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            guesses = json.loads(raw)
            # Only return guesses for valid candidate session IDs
            valid_ids = set(all_ids)
            return [
                {"session_id": g["session_id"], "confidence": float(g["confidence"])}
                for g in guesses
                if isinstance(g, dict) and "session_id" in g
                and g["session_id"] in valid_ids
            ]
        except (json.JSONDecodeError, KeyError):
            return []


class ReidentificationTest:
    """Runs re-identification attacks against the aggregation pipeline."""

    def __init__(
        self,
        summarizer: BaseSummarizer,
        aggregator: BaseAggregator,
        adversary: BaseAdversary,
    ) -> None:
        self.summarizer = summarizer
        self.aggregator = aggregator
        self.adversary = adversary

    async def run_attack(
        self,
        target_traces: list[SessionTrace],
        decoy_traces: list[SessionTrace],
        report: AggregateReport,
    ) -> ReidentificationResult:
        """Run a single re-identification attack."""
        # Summarize all candidates
        all_traces = target_traces + decoy_traces
        summaries = await self.summarizer.summarize_batch(all_traces)

        target_ids = {t.session_id for t in target_traces}
        num_targets = len(target_ids)
        num_decoys = len(decoy_traces)

        # Run adversary attack
        guesses = await self.adversary.attack(summaries, report, num_targets)
        guessed_ids = {g["session_id"] for g in guesses}

        # Compute metrics
        true_positives = len(guessed_ids & target_ids)
        false_positives = len(guessed_ids - target_ids)

        tpr = true_positives / max(num_targets, 1)
        fpr = false_positives / max(num_decoys, 1)
        precision = true_positives / max(len(guessed_ids), 1)
        recall = tpr
        denom = max(precision + recall, 1e-10)
        f1 = (2 * precision * recall / denom) if (precision + recall) > 0 else 0.0

        confidences = [float(g["confidence"]) for g in guesses] if guesses else [0.0]
        mean_confidence = sum(confidences) / len(confidences)

        random_baseline = num_targets / max(num_targets + num_decoys, 1)

        return ReidentificationResult(
            batch_size=num_targets,
            num_targets=num_targets,
            num_decoys=num_decoys,
            true_positive_rate=round(tpr, 4),
            false_positive_rate=round(fpr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            adversary_confidence_mean=round(mean_confidence, 4),
            random_baseline_tpr=round(random_baseline, 4),
        )

    async def run_batch_size_sweep(
        self,
        all_traces: list[SessionTrace],
        batch_sizes: list[int] | None = None,
        num_decoys: int = 20,
        trials_per_batch: int = 3,
        seed: int = 42,
    ) -> list[ReidentificationResult]:
        """Sweep across batch sizes, averaging results across trials."""
        if batch_sizes is None:
            batch_sizes = [5, 10, 20, 50, 100]

        rng = random.Random(seed)
        results: list[ReidentificationResult] = []

        for batch_size in batch_sizes:
            if batch_size + num_decoys > len(all_traces):
                continue

            trial_results: list[ReidentificationResult] = []

            for _ in range(trials_per_batch):
                # Sample target and decoy traces
                sampled = rng.sample(all_traces, batch_size + num_decoys)
                targets = sampled[:batch_size]
                decoys = sampled[batch_size:]

                # Run pipeline on targets only
                target_summaries = await self.summarizer.summarize_batch(targets)
                report = await self.aggregator.aggregate(target_summaries)

                result = await self.run_attack(targets, decoys, report)
                trial_results.append(result)

            # Average across trials
            avg_result = ReidentificationResult(
                batch_size=batch_size,
                num_targets=batch_size,
                num_decoys=num_decoys,
                true_positive_rate=round(
                    sum(r.true_positive_rate for r in trial_results) / len(trial_results), 4
                ),
                false_positive_rate=round(
                    sum(r.false_positive_rate for r in trial_results) / len(trial_results), 4
                ),
                precision=round(
                    sum(r.precision for r in trial_results) / len(trial_results), 4
                ),
                recall=round(
                    sum(r.recall for r in trial_results) / len(trial_results), 4
                ),
                f1=round(
                    sum(r.f1 for r in trial_results) / len(trial_results), 4
                ),
                adversary_confidence_mean=round(
                    sum(r.adversary_confidence_mean for r in trial_results) / len(trial_results), 4
                ),
                random_baseline_tpr=round(
                    batch_size / (batch_size + num_decoys), 4
                ),
            )
            results.append(avg_result)

        return results
