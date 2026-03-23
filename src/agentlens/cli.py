"""Command-line interface for AgentLens.

Usage:
    python -m agentlens summarize --traces-dir ./traces --output ./summaries
    python -m agentlens aggregate --summaries-dir ./summaries --output ./reports
    python -m agentlens run --traces-dir ./traces --output ./reports
    python -m agentlens run --traces-dir ./traces --mock  # No API calls
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentlens.aggregation.aggregator import MockAggregator, SessionAggregator
from agentlens.aggregation.models import SessionSummary
from agentlens.aggregation.pipeline import AgentLensPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentlens",
        description="AgentLens — Privacy-preserving observability for LLM agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- summarize ---
    sp_summarize = subparsers.add_parser(
        "summarize", help="Stage 1: Summarize traces into session summaries"
    )
    sp_summarize.add_argument(
        "--traces-dir", default="./traces", help="Directory containing trace files"
    )
    sp_summarize.add_argument(
        "--output", default="./summaries", help="Output directory for summaries"
    )
    sp_summarize.add_argument(
        "--mock", action="store_true", help="Use MockSummarizer (no API calls)"
    )
    sp_summarize.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Max concurrent API calls (use 1 for Bedrock)",
    )
    sp_summarize.add_argument(
        "--aws-region", default=None, help="AWS region for Bedrock (e.g. us-east-1)"
    )
    sp_summarize.add_argument(
        "--model", default=None,
        help="Model ID (e.g. claude-haiku-4-5-20251001 or Bedrock cross-region inference ID)",
    )

    # --- aggregate ---
    sp_aggregate = subparsers.add_parser(
        "aggregate", help="Stage 2: Aggregate summaries into a report"
    )
    sp_aggregate.add_argument(
        "--summaries-dir", default="./summaries",
        help="Directory containing summary JSON files",
    )
    sp_aggregate.add_argument(
        "--output", default="./reports", help="Output directory for report"
    )
    sp_aggregate.add_argument(
        "--mock", action="store_true", help="Use MockAggregator (no API calls)"
    )
    sp_aggregate.add_argument(
        "--aws-region", default=None, help="AWS region for Bedrock (e.g. us-east-1)"
    )
    sp_aggregate.add_argument(
        "--model", default=None,
        help="Model ID (e.g. claude-haiku-4-5-20251001 or Bedrock cross-region inference ID)",
    )

    # --- run ---
    sp_run = subparsers.add_parser(
        "run", help="Run full pipeline: traces -> summaries -> report"
    )
    sp_run.add_argument(
        "--traces-dir", default="./traces", help="Directory containing trace files"
    )
    sp_run.add_argument(
        "--output", default="./reports", help="Output directory for reports"
    )
    sp_run.add_argument(
        "--mock", action="store_true", help="Use mock components (no API calls)"
    )
    sp_run.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Max concurrent API calls (use 1 for Bedrock)",
    )
    sp_run.add_argument(
        "--aws-region", default=None, help="AWS region for Bedrock (e.g. us-east-1)"
    )
    sp_run.add_argument(
        "--model", default=None,
        help="Model ID (e.g. claude-haiku-4-5-20251001 or Bedrock cross-region inference ID)",
    )

    # --- analyze ---
    sp_analyze = subparsers.add_parser(
        "analyze", help="Run five-dimensional oversight analysis on session summaries"
    )
    sp_analyze.add_argument(
        "--summaries-dir", default="./summaries",
        help="Directory containing summary JSON files",
    )
    sp_analyze.add_argument(
        "--output", default="./analysis_results",
        help="Output directory for results, report, and optional plots",
    )
    sp_analyze.add_argument(
        "--plots", action="store_true", help="Also generate matplotlib plots"
    )

    return parser


async def _cmd_summarize(args: argparse.Namespace) -> None:
    model_kwargs = {}
    if args.model:
        model_kwargs["summarizer_model"] = args.model
    pipeline = AgentLensPipeline(
        traces_dir=args.traces_dir,
        summaries_dir=args.output,
        use_mock=args.mock,
        aws_region=args.aws_region,
        **model_kwargs,
    )
    traces = pipeline.load_traces()
    if not traces:
        print(f"No traces found in {args.traces_dir}", file=sys.stderr)
        sys.exit(1)
    summaries = await pipeline.run_stage1(traces, max_concurrent=args.max_concurrency)
    print(f"Summarized {len(summaries)} traces -> {args.output}")


async def _cmd_aggregate(args: argparse.Namespace) -> None:
    summaries_dir = Path(args.summaries_dir)
    if not summaries_dir.exists():
        print(f"Summaries directory not found: {args.summaries_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    skipped = 0
    for p in summaries_dir.glob("*.json"):
        text = p.read_text().strip()
        if text:
            try:
                summaries.append(SessionSummary.from_json(text))
            except Exception as exc:
                print(f"Warning: skipping {p.name}: {exc}", file=sys.stderr)
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} invalid summary file(s)", file=sys.stderr)

    if not summaries:
        print(f"No summaries found in {args.summaries_dir}", file=sys.stderr)
        sys.exit(1)

    if args.mock:
        aggregator = MockAggregator()
    else:
        agg_kwargs = {"aws_region": args.aws_region}
        if args.model:
            agg_kwargs["model"] = args.model
        aggregator = SessionAggregator(**agg_kwargs)

    report = await aggregator.aggregate(summaries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report.report_id}.json"
    report_path.write_text(report.to_json())
    print(f"Aggregated {len(summaries)} summaries -> {report_path}")


async def _cmd_run(args: argparse.Namespace) -> None:
    model_kwargs = {}
    if args.model:
        model_kwargs["summarizer_model"] = args.model
        model_kwargs["aggregator_model"] = args.model
    pipeline = AgentLensPipeline(
        traces_dir=args.traces_dir,
        reports_dir=args.output,
        use_mock=args.mock,
        aws_region=args.aws_region,
        **model_kwargs,
    )
    report = await pipeline.run(max_concurrent=args.max_concurrency)
    print(f"Pipeline complete. Report: {args.output}/{report.report_id}.json")


def _cmd_analyze(args: argparse.Namespace) -> None:
    from agentlens.analysis.analyzer import AgentAnalyzer
    from agentlens.analysis.report import generate_analysis_report

    summaries_dir = Path(args.summaries_dir)
    if not summaries_dir.exists():
        print(f"Summaries directory not found: {args.summaries_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    skipped = 0
    for p in summaries_dir.glob("*.json"):
        text = p.read_text().strip()
        if text:
            try:
                summaries.append(SessionSummary.from_json(text))
            except Exception as exc:
                print(f"Warning: skipping {p.name}: {exc}", file=sys.stderr)
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} invalid summary file(s)", file=sys.stderr)

    if not summaries:
        print(f"No summaries found in {args.summaries_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(summaries)} summaries")
    analyzer = AgentAnalyzer.__new__(AgentAnalyzer)
    analyzer.summaries = summaries

    results = analyzer.run_all()
    analyzer.save_results(results, args.output)
    print(f"Results saved to {args.output}/results.json")

    report_path = f"{args.output}/analysis_report.md"
    generate_analysis_report(results, report_path)
    print(f"Report saved to {report_path}")

    if args.plots:
        analyzer.generate_plots(results, args.output)
        print(f"Plots saved to {args.output}/plots/")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _cmd_analyze(args)
        return

    commands = {
        "summarize": _cmd_summarize,
        "aggregate": _cmd_aggregate,
        "run": _cmd_run,
    }
    asyncio.run(commands[args.command](args))
