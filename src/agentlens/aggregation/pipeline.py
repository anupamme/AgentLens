"""End-to-end pipeline: raw traces -> session summaries -> aggregate report.

Usage:
    pipeline = AgentLensPipeline(api_key="...", traces_dir="./traces")
    report = await pipeline.run()
    pipeline.save_report(report, "./reports/weekly_report.json")
"""

from __future__ import annotations

import json
from pathlib import Path

from agentlens.aggregation.aggregator import BaseAggregator, MockAggregator, SessionAggregator
from agentlens.aggregation.models import AggregateReport, SessionSummary
from agentlens.aggregation.summarizer import BaseSummarizer, MockSummarizer, SessionSummarizer
from agentlens.schema.trace import SessionTrace


class AgentLensPipeline:
    """End-to-end pipeline: raw traces -> session summaries -> aggregate report."""

    def __init__(
        self,
        api_key: str | None = None,
        traces_dir: str = "./traces",
        summaries_dir: str = "./summaries",
        reports_dir: str = "./reports",
        use_mock: bool = False,
        summarizer_model: str = "claude-haiku-4-5-20251001",
        aggregator_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.summaries_dir = Path(summaries_dir)
        self.reports_dir = Path(reports_dir)

        if use_mock:
            self.summarizer: BaseSummarizer = MockSummarizer()
            self.aggregator: BaseAggregator = MockAggregator()
        else:
            self.summarizer = SessionSummarizer(
                api_key=api_key, model=summarizer_model
            )
            self.aggregator = SessionAggregator(
                api_key=api_key, model=aggregator_model
            )

    def load_traces(self) -> list[SessionTrace]:
        """Load all traces from traces_dir (JSONL and JSON files)."""
        if not self.traces_dir.exists():
            return []

        traces: list[SessionTrace] = []

        # Load JSONL files
        for jsonl_path in self.traces_dir.glob("**/*.jsonl"):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(SessionTrace.from_json(line))

        # Load individual JSON files
        for json_path in self.traces_dir.glob("**/*.json"):
            text = json_path.read_text().strip()
            if text:
                traces.append(SessionTrace.from_json(text))

        return traces

    async def run_stage1(
        self, traces: list[SessionTrace], max_concurrent: int = 5
    ) -> list[SessionSummary]:
        """Stage 1: Summarize each trace. Save summaries to summaries_dir."""
        summaries = await self.summarizer.summarize_batch(
            traces, max_concurrent=max_concurrent
        )

        # Save summaries
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        for summary in summaries:
            path = self.summaries_dir / f"{summary.session_id}.json"
            path.write_text(summary.to_json())

        return summaries

    async def run_stage2(
        self, summaries: list[SessionSummary]
    ) -> AggregateReport:
        """Stage 2: Aggregate summaries into a report. Save to reports_dir."""
        report = await self.aggregator.aggregate(summaries)

        # Save report
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{report.report_id}.json"
        path.write_text(report.to_json())

        return report

    async def run(self, max_concurrent: int = 5) -> AggregateReport:
        """Run the full pipeline end-to-end."""
        traces = self.load_traces()
        if not traces:
            raise ValueError(
                f"No traces found in {self.traces_dir}. "
                f"Ensure trace files (.json or .jsonl) exist in the directory."
            )
        summaries = await self.run_stage1(traces, max_concurrent=max_concurrent)
        report = await self.run_stage2(summaries)
        return report

    def save_report(self, report: AggregateReport, path: str) -> None:
        """Save report as JSON to a specific path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report.to_json())
