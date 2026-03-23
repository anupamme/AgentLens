"""CLI for workload generation and data collection.

Usage:
    python -m agentlens.workloads generate --agent-type code_reviewer --count 50 --mock
    python -m agentlens.workloads run --workloads-dir ./workloads --output-dir ./traces
    python -m agentlens.workloads validate --traces-dir ./traces
    python -m agentlens.workloads campaign --mock --counts '{"code_reviewer": 50}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agentlens.workloads.generator import TaskConfig
from agentlens.workloads.mock_generator import MockWorkloadGenerator
from agentlens.workloads.runner import WorkloadRunner
from agentlens.workloads.validator import TraceValidator


def _load_traces_from_dir(traces_dir: Path) -> list[Any]:
    """Load traces from JSONL files, skipping non-trace JSON files."""
    from agentlens.schema.trace import SessionTrace

    traces = []
    for jsonl_path in traces_dir.glob("**/*.jsonl"):
        text = jsonl_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if line:
                traces.append(SessionTrace.from_json(line))
    return traces


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentlens.workloads",
        description="AgentLens workload generation and data collection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    sp_gen = subparsers.add_parser(
        "generate", help="Generate workload task configurations"
    )
    sp_gen.add_argument("--agent-type", required=True, help="Agent type to generate for")
    sp_gen.add_argument("--count", type=int, default=50, help="Number of tasks to generate")
    sp_gen.add_argument("--output", default="./workloads", help="Output directory")
    sp_gen.add_argument("--mock", action="store_true", help="Use mock generator (no LLM)")
    sp_gen.add_argument("--seed", type=int, default=None, help="Random seed for mock generator")
    sp_gen.add_argument("--aws-region", default=None, help="AWS region for Bedrock (e.g. us-east-1)")
    sp_gen.add_argument("--model", default=None, help="Model ID (e.g. us.anthropic.claude-sonnet-4-20250514-v1:0 for Bedrock)")

    # --- run ---
    sp_run = subparsers.add_parser(
        "run", help="Run workloads and collect traces"
    )
    sp_run.add_argument("--workloads-dir", required=True, help="Directory with workload JSON files")
    sp_run.add_argument("--output-dir", default="./traces", help="Output directory for traces")
    sp_run.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    sp_run.add_argument("--budget-limit", type=float, default=None, help="Max cost budget (unused for simulated)")

    # --- validate ---
    sp_val = subparsers.add_parser(
        "validate", help="Validate collected traces"
    )
    sp_val.add_argument("--traces-dir", required=True, help="Directory with trace files")

    # --- campaign ---
    sp_camp = subparsers.add_parser(
        "campaign", help="Run a full generation + execution + validation campaign"
    )
    sp_camp.add_argument(
        "--counts", required=True,
        help='JSON mapping agent_type to count, e.g. \'{"code_reviewer": 50}\'',
    )
    sp_camp.add_argument("--output-dir", default="./traces/workloads", help="Output directory")
    sp_camp.add_argument("--mock", action="store_true", help="Use mock generator (no LLM)")
    sp_camp.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    sp_camp.add_argument("--seed", type=int, default=None, help="Random seed for mock generator")
    sp_camp.add_argument("--aws-region", default=None, help="AWS region for Bedrock (e.g. us-east-1)")
    sp_camp.add_argument("--model", default=None, help="Model ID (e.g. us.anthropic.claude-sonnet-4-20250514-v1:0 for Bedrock)")

    return parser


async def _cmd_generate(args: argparse.Namespace) -> None:
    tasks: list[TaskConfig]
    if args.mock:
        mock_gen = MockWorkloadGenerator(seed=args.seed)
        tasks = mock_gen.generate(agent_type=args.agent_type, count=args.count)
    else:
        from agentlens.workloads.generator import WorkloadGenerator
        gen_kwargs: dict[str, Any] = {}
        if args.aws_region:
            gen_kwargs["aws_region"] = args.aws_region
        if args.model:
            gen_kwargs["model"] = args.model
        llm_gen = WorkloadGenerator(**gen_kwargs)
        tasks = await llm_gen.generate(agent_type=args.agent_type, count=args.count)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.agent_type}.json"
    output_path.write_text(
        json.dumps([t.model_dump() for t in tasks], indent=2)
    )
    print(f"Generated {len(tasks)} tasks -> {output_path}")


async def _cmd_run(args: argparse.Namespace) -> None:
    workloads_dir = Path(args.workloads_dir)
    if not workloads_dir.exists():
        print(f"Workloads directory not found: {args.workloads_dir}", file=sys.stderr)
        sys.exit(1)

    workloads: dict[str, list[TaskConfig]] = {}
    for p in workloads_dir.glob("*.json"):
        items = json.loads(p.read_text())
        tasks = [TaskConfig(**item) for item in items]
        if tasks:
            agent_type = tasks[0].agent_type
            workloads.setdefault(agent_type, []).extend(tasks)

    if not workloads:
        print("No workload files found", file=sys.stderr)
        sys.exit(1)

    runner = WorkloadRunner(output_dir=args.output_dir)
    results = await runner.run_all(workloads, max_concurrent=args.max_concurrent)

    total = sum(len(r) for r in results.values())
    successes = sum(1 for rs in results.values() for r in rs if r.success)
    print(f"Ran {total} tasks: {successes} success, {total - successes} failures")

    output_parent = Path(args.output_dir).parent
    report_path = output_parent / "run_report.json"
    WorkloadRunner.save_run_report(results, str(report_path))
    print(f"Report saved to {report_path}")


async def _cmd_validate(args: argparse.Namespace) -> None:
    from agentlens.schema.trace import SessionTrace

    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"Traces directory not found: {args.traces_dir}", file=sys.stderr)
        sys.exit(1)

    traces = _load_traces_from_dir(traces_dir)

    if not traces:
        print(f"No traces found in {args.traces_dir}", file=sys.stderr)
        sys.exit(1)

    validator = TraceValidator()
    report = validator.validate_batch(traces)

    print(f"Validation Report:")
    print(f"  Total traces:    {report.total_traces}")
    print(f"  Valid traces:    {report.valid_traces}")
    print(f"  Validation rate: {report.validation_rate:.1%}")
    print(f"  Diversity score: {report.diversity_score:.2f}")
    if report.errors_by_type:
        print(f"  Errors:")
        for err_type, count in sorted(report.errors_by_type.items()):
            print(f"    {err_type}: {count}")
    if report.warnings:
        print(f"  Warnings:")
        for w in report.warnings:
            print(f"    {w}")
    print(f"  Agent distribution: {report.agent_distribution}")


async def _cmd_campaign(args: argparse.Namespace) -> None:
    counts = json.loads(args.counts)

    # Step 1: Generate
    print("Step 1: Generating workloads...")
    mock_gen2 = MockWorkloadGenerator(seed=args.seed) if args.mock else None
    all_workloads: dict[str, list[TaskConfig]] = {}

    for agent_type, count in counts.items():
        if args.mock and mock_gen2 is not None:
            tasks = mock_gen2.generate(agent_type=agent_type, count=count)
        else:
            from agentlens.workloads.generator import WorkloadGenerator
            gen_kwargs: dict[str, Any] = {}
            if args.aws_region:
                gen_kwargs["aws_region"] = args.aws_region
            if args.model:
                gen_kwargs["model"] = args.model
            llm_gen = WorkloadGenerator(**gen_kwargs)
            tasks = await llm_gen.generate(agent_type=agent_type, count=count)
        all_workloads[agent_type] = tasks
        print(f"  {agent_type}: {len(tasks)} tasks")

    # Step 2: Run
    print("Step 2: Running workloads...")
    runner = WorkloadRunner(output_dir=args.output_dir)
    results = await runner.run_all(all_workloads, max_concurrent=args.max_concurrent)

    total = sum(len(r) for r in results.values())
    successes = sum(1 for rs in results.values() for r in rs if r.success)
    print(f"  Completed: {successes}/{total} successful")

    # Step 3: Validate
    print("Step 3: Validating traces...")
    traces = _load_traces_from_dir(Path(args.output_dir))

    validator = TraceValidator()
    report = validator.validate_batch(traces)

    print(f"  Validation rate: {report.validation_rate:.1%}")
    print(f"  Diversity score: {report.diversity_score:.2f}")

    # Save reports alongside (not inside) the traces dir to avoid interfering
    # with pipeline.load_traces() which globs **/*.json
    output_parent = Path(args.output_dir).parent
    report_path = output_parent / "campaign_report.json"
    report_path.write_text(report.model_dump_json(indent=2))
    WorkloadRunner.save_run_report(
        results, str(output_parent / "run_report.json")
    )

    print(f"\nCampaign complete. Traces in {args.output_dir}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": _cmd_generate,
        "run": _cmd_run,
        "validate": _cmd_validate,
        "campaign": _cmd_campaign,
    }
    asyncio.run(commands[args.command](args))
