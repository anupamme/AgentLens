"""AgentAnalyzer orchestrator — runs all five dimensions of analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentlens.aggregation.models import SessionSummary
from agentlens.analysis.autonomy import analyze_autonomy
from agentlens.analysis.escalations import analyze_escalations
from agentlens.analysis.failures import analyze_failures
from agentlens.analysis.models import AnalysisResults
from agentlens.analysis.oversight_gap import analyze_oversight_gap
from agentlens.analysis.tools import analyze_tool_usage


class AgentAnalyzer:
    """Orchestrates all five dimensions of agent oversight analysis."""

    def __init__(self, summaries_dir: str = "./summaries") -> None:
        self.summaries = self._load_summaries(summaries_dir)

    def _load_summaries(self, path: str) -> list[SessionSummary]:
        """Load SessionSummary objects from *.json and *.jsonl files."""
        summaries: list[SessionSummary] = []
        summaries_dir = Path(path)
        if not summaries_dir.exists():
            return summaries

        # Load .json files (one summary per file)
        for p in summaries_dir.glob("*.json"):
            text = p.read_text().strip()
            if text:
                try:
                    summaries.append(SessionSummary.from_json(text))
                except Exception:
                    pass

        # Load .jsonl files (one summary per line)
        for p in summaries_dir.glob("*.jsonl"):
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        summaries.append(SessionSummary.from_json(line))
                    except Exception:
                        pass

        return summaries

    def run_all(self) -> AnalysisResults:
        """Run all five analysis dimensions and return combined results."""
        return AnalysisResults(
            autonomy=analyze_autonomy(self.summaries),
            failures=analyze_failures(self.summaries),
            tools=analyze_tool_usage(self.summaries),
            escalations=analyze_escalations(self.summaries),
            oversight_gap=analyze_oversight_gap(self.summaries),
            metadata={
                "session_count": len(self.summaries),
                "agent_types": sorted(set(s.agent_type for s in self.summaries)),
                "task_categories": sorted(set(s.task_category.value for s in self.summaries)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def save_results(self, results: AnalysisResults, output_dir: str) -> None:
        """Save analysis results as JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(results.to_json())

    def generate_plots(self, results: AnalysisResults, output_dir: str) -> None:
        """Generate all plots using matplotlib."""
        from agentlens.analysis.plots import plot_all

        plot_all(results, output_dir)
