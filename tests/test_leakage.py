"""Tests for PII leakage detection."""

import pytest

from agentlens.aggregation.aggregator import MockAggregator
from agentlens.aggregation.summarizer import MockSummarizer
from agentlens.privacy.leakage_test import (
    PIILeakageReport,
    PIILeakageTest,
    _levenshtein_distance,
    check_text_for_pii,
)
from agentlens.privacy.pii_generator import PIIGenerator


class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_single_insertion(self):
        assert _levenshtein_distance("hello", "helllo") == 1

    def test_single_deletion(self):
        assert _levenshtein_distance("hello", "helo") == 1

    def test_single_substitution(self):
        assert _levenshtein_distance("hello", "hallo") == 1

    def test_completely_different(self):
        assert _levenshtein_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_known_pairs(self):
        assert _levenshtein_distance("kitten", "sitting") == 3
        assert _levenshtein_distance("saturday", "sunday") == 3


class TestCheckTextForPII:
    def test_exact_match_detection(self):
        pii = {"john.doe@example.com", "555-123-4567"}
        text = "Contact john.doe@example.com for details"
        detected = check_text_for_pii(text, pii)
        assert "john.doe@example.com" in detected

    def test_case_insensitive_match(self):
        pii = {"John.Doe@Example.com"}
        text = "contact JOHN.DOE@EXAMPLE.COM for info"
        detected = check_text_for_pii(text, pii)
        assert "John.Doe@Example.com" in detected

    def test_fuzzy_match_detection(self):
        pii = {"john.doe@example.com"}
        # One character different (edit distance 1)
        text = "contact john.doa@example.com for info"
        detected = check_text_for_pii(text, pii)
        assert "john.doe@example.com" in detected

    def test_no_false_positives_on_clean_text(self):
        pii = {"john.doe@example.com", "555-123-4567", "sk-abc123def456"}
        text = (
            "Agent performed 5 actions on a code review task. "
            "Tools used: search_tool, write_tool. "
            "Success rate: 80%. Duration: 5.2 seconds."
        )
        detected = check_text_for_pii(text, pii)
        assert detected == []

    def test_skip_fuzzy_for_short_strings(self):
        pii = {"Jo", "abc"}
        text = "Jo went to the store and bought abc items"
        detected = check_text_for_pii(text, pii)
        # Exact matches should work, fuzzy should be skipped for short strings
        assert "Jo" in detected
        assert "abc" in detected

    def test_empty_text(self):
        pii = {"test@example.com"}
        assert check_text_for_pii("", pii) == []

    def test_empty_pii_set(self):
        assert check_text_for_pii("some text here", set()) == []


class TestPIILeakageTestIntegration:
    @pytest.mark.asyncio
    async def test_full_mock_pipeline(self):
        """Run full leakage test with mock components."""
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        pii_gen = PIIGenerator(seed=42)

        test = PIILeakageTest(summarizer, aggregator, pii_gen)
        report = test.run(num_traces=10)
        report = await report

        assert isinstance(report, PIILeakageReport)
        assert report.num_traces_tested == 10

        # Stage 1: MockSummarizer generates generic abstracts, shouldn't leak
        assert isinstance(report.stage1_leakage_rate, float)
        assert 0.0 <= report.stage1_leakage_rate <= 1.0

        # Stage 2: MockAggregator produces generic narrative, should be zero
        assert report.stage2_leakage_rate == 0.0

        # Overall pass should be True (stage2 is zero)
        assert report.overall_pass is True

    @pytest.mark.asyncio
    async def test_report_structure(self):
        """Verify report has all expected fields."""
        summarizer = MockSummarizer()
        aggregator = MockAggregator()
        pii_gen = PIIGenerator(seed=99)

        test = PIILeakageTest(summarizer, aggregator, pii_gen)
        report = await test.run(num_traces=5)

        assert hasattr(report, "num_traces_tested")
        assert hasattr(report, "stage1_leakage_rate")
        assert hasattr(report, "stage1_leaked_types")
        assert hasattr(report, "stage2_leakage_rate")
        assert hasattr(report, "overall_pass")
