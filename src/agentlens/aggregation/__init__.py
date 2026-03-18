"""AgentLens aggregation pipeline — privacy-preserving trace analysis."""

from agentlens.aggregation.models import AggregateReport, SessionSummary
from agentlens.aggregation.summarizer import BaseSummarizer, MockSummarizer, SessionSummarizer
from agentlens.aggregation.aggregator import (
    BaseAggregator,
    MockAggregator,
    SessionAggregator,
    compute_statistics,
)
from agentlens.aggregation.pipeline import AgentLensPipeline

__all__ = [
    "AggregateReport",
    "AgentLensPipeline",
    "BaseAggregator",
    "BaseSummarizer",
    "MockAggregator",
    "MockSummarizer",
    "SessionAggregator",
    "SessionSummary",
    "SessionSummarizer",
    "compute_statistics",
]
