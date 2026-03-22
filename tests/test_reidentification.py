"""Tests for re-identification attack experiments."""

import pytest

from agentlens.aggregation.aggregator import MockAggregator
from agentlens.aggregation.summarizer import MockSummarizer
from agentlens.privacy.reidentification_test import (
    MockAdversary,
    ReidentificationResult,
    ReidentificationTest,
)
from agentlens.privacy.trace_factory import make_diverse_traces


class TestMockAdversary:
    @pytest.mark.asyncio
    async def test_returns_valid_session_ids(self):
        summarizer = MockSummarizer()
        traces = make_diverse_traces(count=10, seed=42)
        summaries = await summarizer.summarize_batch(traces)

        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        adversary = MockAdversary(seed=42)
        guesses = await adversary.attack(summaries, report, num_targets=5)

        valid_ids = {s.session_id for s in summaries}
        for g in guesses:
            assert g["session_id"] in valid_ids
            assert 0.0 <= g["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_deterministic_with_same_seed(self):
        summarizer = MockSummarizer()
        traces = make_diverse_traces(count=10, seed=42)
        summaries = await summarizer.summarize_batch(traces)

        aggregator = MockAggregator()
        report = await aggregator.aggregate(summaries)

        adv1 = MockAdversary(seed=123)
        adv2 = MockAdversary(seed=123)

        g1 = await adv1.attack(summaries, report, num_targets=5)
        g2 = await adv2.attack(summaries, report, num_targets=5)

        assert [g["session_id"] for g in g1] == [g["session_id"] for g in g2]


class TestReidentificationTest:
    @pytest.mark.asyncio
    async def test_run_attack_result_fields(self):
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        adversary = MockAdversary(seed=42)

        traces = make_diverse_traces(count=15, seed=42)
        targets = traces[:5]
        decoys = traces[5:15]

        target_summaries = await summarizer.summarize_batch(targets)
        report = await aggregator.aggregate(target_summaries)

        test = ReidentificationTest(summarizer, aggregator, adversary)
        result = await test.run_attack(targets, decoys, report)

        assert isinstance(result, ReidentificationResult)
        assert 0.0 <= result.true_positive_rate <= 1.0
        assert 0.0 <= result.false_positive_rate <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert result.num_targets == 5
        assert result.num_decoys == 10

    @pytest.mark.asyncio
    async def test_random_baseline_calculation(self):
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        adversary = MockAdversary(seed=42)

        traces = make_diverse_traces(count=30, seed=42)
        targets = traces[:10]
        decoys = traces[10:30]

        target_summaries = await summarizer.summarize_batch(targets)
        report = await aggregator.aggregate(target_summaries)

        test = ReidentificationTest(summarizer, aggregator, adversary)
        result = await test.run_attack(targets, decoys, report)

        expected_baseline = 10 / (10 + 20)
        assert abs(result.random_baseline_tpr - expected_baseline) < 0.001

    @pytest.mark.asyncio
    async def test_batch_size_sweep(self):
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        adversary = MockAdversary(seed=42)

        traces = make_diverse_traces(count=50, seed=42)

        test = ReidentificationTest(summarizer, aggregator, adversary)
        results = await test.run_batch_size_sweep(
            all_traces=traces,
            batch_sizes=[5, 10, 20],
            num_decoys=10,
            trials_per_batch=2,
            seed=42,
        )

        assert len(results) == 3
        assert results[0].batch_size == 5
        assert results[1].batch_size == 10
        assert results[2].batch_size == 20

        for r in results:
            assert 0.0 <= r.true_positive_rate <= 1.0
            assert 0.0 <= r.false_positive_rate <= 1.0

    @pytest.mark.asyncio
    async def test_sweep_skips_oversized_batches(self):
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        adversary = MockAdversary(seed=42)

        traces = make_diverse_traces(count=25, seed=42)

        test = ReidentificationTest(summarizer, aggregator, adversary)
        results = await test.run_batch_size_sweep(
            all_traces=traces,
            batch_sizes=[5, 10, 50, 100],  # 50 and 100 won't fit
            num_decoys=20,
            trials_per_batch=1,
            seed=42,
        )

        # Only batch_size=5 fits (5+20=25)
        assert len(results) == 1
        assert results[0].batch_size == 5
