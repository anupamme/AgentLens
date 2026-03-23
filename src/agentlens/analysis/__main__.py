"""Entry point for `python -m agentlens.analysis`."""

from __future__ import annotations

import argparse
import sys

from agentlens.analysis.analyzer import AgentAnalyzer
from agentlens.analysis.models import AnalysisResults
from agentlens.analysis.report import generate_analysis_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentlens.analysis",
        description="Five-dimensional agent oversight analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    sp_run = subparsers.add_parser("run", help="Run analysis on session summaries")
    sp_run.add_argument(
        "--summaries-dir", default="./summaries",
        help="Directory containing summary JSON/JSONL files",
    )
    sp_run.add_argument(
        "--output", default="./analysis_results",
        help="Output directory for results",
    )
    sp_run.add_argument(
        "--plots", action="store_true",
        help="Also generate matplotlib plots",
    )

    # --- report ---
    sp_report = subparsers.add_parser("report", help="Generate markdown report from results")
    sp_report.add_argument(
        "--results-dir", default="./analysis_results",
        help="Directory containing results.json",
    )
    sp_report.add_argument(
        "--output", default="./analysis_report.md",
        help="Output path for markdown report",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        analyzer = AgentAnalyzer(summaries_dir=args.summaries_dir)
        if not analyzer.summaries:
            print(f"No summaries found in {args.summaries_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(analyzer.summaries)} summaries")

        results = analyzer.run_all()
        analyzer.save_results(results, args.output)
        print(f"Results saved to {args.output}/results.json")

        # Generate report
        report_path = f"{args.output}/analysis_report.md"
        generate_analysis_report(results, report_path)
        print(f"Report saved to {report_path}")

        if args.plots:
            analyzer.generate_plots(results, args.output)
            print(f"Plots saved to {args.output}/plots/")

    elif args.command == "report":
        from pathlib import Path

        results_path = Path(args.results_dir) / "results.json"
        if not results_path.exists():
            print(f"Results file not found: {results_path}", file=sys.stderr)
            sys.exit(1)

        results = AnalysisResults.from_json(results_path.read_text())
        generate_analysis_report(results, args.output)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
