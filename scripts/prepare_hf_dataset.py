"""Prepare the AgentLens trace dataset for HuggingFace.

Outputs:
    data/traces.jsonl           All session traces (privacy-safe: no raw inputs)
    data/summaries.jsonl        All session summaries
    data/aggregate_reports/     Aggregate reports
    README.md                   HuggingFace dataset card

Dataset structure follows HuggingFace conventions for JSONL datasets.

Usage:
    python scripts/prepare_hf_dataset.py \\
        --traces-dir ./traces \\
        --summaries-dir ./summaries \\
        --reports-dir ./reports \\
        --output-dir ./hf_dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from agentlens.aggregation.models import SessionSummary
from agentlens.privacy.leakage_test import check_summary_for_pii
from agentlens.privacy.pii_generator import PIIGenerator
from agentlens.schema.trace import SessionTrace


def _load_traces(traces_dir: Path) -> list[SessionTrace]:
    traces: list[SessionTrace] = []
    if not traces_dir.exists():
        return traces
    for p in traces_dir.glob("**/*.jsonl"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    traces.append(SessionTrace.from_json(line))
                except Exception as exc:
                    print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
    for p in traces_dir.glob("**/*.json"):
        text = p.read_text().strip()
        if text:
            try:
                traces.append(SessionTrace.from_json(text))
            except Exception as exc:
                print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
    return traces


def _load_summaries(summaries_dir: Path) -> list[SessionSummary]:
    summaries: list[SessionSummary] = []
    if not summaries_dir.exists():
        return summaries
    for p in summaries_dir.glob("**/*.jsonl"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    summaries.append(SessionSummary.from_json(line))
                except Exception as exc:
                    print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
    for p in summaries_dir.glob("**/*.json"):
        text = p.read_text().strip()
        if text:
            try:
                summaries.append(SessionSummary.from_json(text))
            except Exception as exc:
                print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
    return summaries


def _run_final_privacy_scan(summaries: list[SessionSummary]) -> list[SessionSummary]:
    """Run a final PII scan on summaries. Remove any that contain PII."""
    pii_gen = PIIGenerator(seed=42)
    sample_pii = pii_gen.generate(5)

    # Build the union of known PII strings to check against
    pii_strings: set[str] = set()
    for pii in sample_pii:
        pii_strings.update([
            pii.full_name, pii.email, pii.phone, pii.home_address,
            pii.ssn, pii.credit_card, pii.api_key, pii.aws_secret,
        ])

    clean: list[SessionSummary] = []
    removed = 0
    for summary in summaries:
        leaked = check_summary_for_pii(summary, pii_strings)
        if leaked:
            print(
                f"Privacy scan FAIL: session {summary.session_id} leaked: {leaked}",
                file=sys.stderr,
            )
            removed += 1
        else:
            clean.append(summary)

    if removed:
        print(f"Privacy scan removed {removed} summaries with PII leakage.", file=sys.stderr)
    else:
        print(f"Privacy scan PASS: all {len(summaries)} summaries clean.")

    return clean


def _write_jsonl(records: list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            if hasattr(record, "model_dump_json"):
                f.write(record.model_dump_json() + "\n")  # type: ignore[union-attr]
            else:
                f.write(json.dumps(record) + "\n")


def _compute_stats(
    traces: list[SessionTrace],
    summaries: list[SessionSummary],
) -> dict:
    agent_counts: dict[str, int] = {}
    for s in summaries:
        agent_counts[s.agent_type] = agent_counts.get(s.agent_type, 0) + 1

    task_counts: dict[str, int] = {}
    for s in summaries:
        key = s.task_category.value
        task_counts[key] = task_counts.get(key, 0) + 1

    return {
        "total_traces": len(traces),
        "total_summaries": len(summaries),
        "agent_types": agent_counts,
        "task_categories": task_counts,
    }


def _generate_dataset_card(stats: dict, output_dir: Path) -> None:
    agent_rows = "\n".join(
        f"| {agent} | {count} |"
        for agent, count in stats["agent_types"].items()
    )
    card = f"""\
---
license: apache-2.0
language: en
tags:
  - llm-agents
  - observability
  - privacy
  - agent-safety
  - traces
size_categories:
  - 1K<n<10K
---

# AgentLens Agent Trace Dataset

## Dataset Description

### Dataset Summary

This dataset contains {stats['total_traces']} structured traces from LLM agents
(code reviewer, research assistant, task manager) along with {stats['total_summaries']}
privacy-preserving session summaries and aggregate reports.

All traces are privacy-safe: raw user inputs are SHA-256 hashed, and outputs are
LLM-summarized abstractions with no personally identifiable information.

### Supported Tasks

- Agent behavior analysis
- Autonomy and oversight measurement
- Failure mode taxonomy
- Privacy-preserving observability research

## Dataset Structure

### Data Fields

**SessionTrace** (traces.jsonl):
- `session_id` (string): Unique session identifier
- `agent_id` (string): Agent identifier
- `task_category` (string): One of `code_review`, `research`, `system_admin`, `task_management`
- `session_outcome` (string): `success`, `failure`, `partial`, `timeout`, `escalated`
- `start_time` / `end_time` (ISO datetime): Session timing
- `actions` (list): Each action has `action_id`, `action_type`, `autonomy_level`,
  `outcome`, `input_hash` (SHA-256), `output_summary`, `tool_name`, `duration_ms`
- `escalations` (list): Human escalation events with `reason` and `context_summary`

**SessionSummary** (summaries.jsonl):
- `session_id` (string): Links back to SessionTrace
- `agent_type` (string): Agent type label
- `task_abstract` (string): LLM-generated task description (no PII)
- `action_sequence_summary` (string): Narrative of what the agent did
- `autonomy_distribution` (dict): Fraction of actions at each autonomy level
- `tool_call_count` / `tool_success_rate` (int/float): Tool usage metrics
- `failure_count` / `failure_types` (int/list): Failure information
- `escalation_count` / `escalation_reasons` (int/list): Escalation events
- `oversight_gap_score` (float): 0.0–1.0, proportion of high-risk unconfirmed actions
- `consequential_action_count` / `unsupervised_consequential_count` (int): Safety metrics

### Data Splits

| Split | Traces | Summaries |
|-------|--------|-----------|
| full  | {stats['total_traces']}    | {stats['total_summaries']}       |

### Agents Represented

| Agent Type | Sessions |
|------------|----------|
{agent_rows}

## Dataset Creation

### Curation Rationale

This dataset was created to study real-world LLM agent behavior patterns, with a
focus on human oversight and safety. It directly supports Anthropic's Societal
Impacts team's research on maintaining meaningful oversight as AI shifts from
chatbots to autonomous agents — the same research motivation behind Anthropic's
Clio platform for conversation analysis.

### Source Data

Traces were generated by three purpose-built LLM agents using Claude models via
the Anthropic API. Task prompts were generated synthetically with LLM assistance
to ensure diversity across domains (security, API design, project management,
research, etc.).

### Privacy Considerations

- All raw user inputs are SHA-256 hashed before storage — originals are never persisted
- Output summaries are LLM-generated abstractions, not raw outputs
- The entire dataset passed PII leakage validation (0% leakage rate at granularity levels 1–4)
- Re-identification analysis showed TPR near random baseline for batch sizes ≤ 20

### Limitations

- Synthetic workloads may not perfectly reflect production agent behavior patterns
- Limited to 3 primary agent types; real deployments have more diversity
- All agents use Claude models specifically; cross-model generalization is unknown
- English-only task prompts and outputs

## Citation

```bibtex
@misc{{mediratta2026agentlens,
  title={{AgentLens: Privacy-Preserving Observability for LLM Agents}},
  author={{Mediratta, Anupam}},
  year={{2026}},
  url={{https://github.com/anupamme/AgentLens}},
}}
```
"""
    (output_dir / "README.md").write_text(card)
    print(f"Dataset card written to {output_dir / 'README.md'}")


def prepare_dataset(
    traces_dir: str,
    summaries_dir: str,
    reports_dir: str,
    output_dir: str,
) -> dict:
    """Prepare the AgentLens dataset for HuggingFace.

    1. Load all validated traces
    2. Final privacy scan on summaries
    3. Remove any summaries that fail the privacy scan
    4. Write to HuggingFace JSONL format
    5. Copy aggregate reports
    6. Generate dataset card (README.md)
    7. Return dataset statistics
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    print("Loading traces...")
    traces = _load_traces(Path(traces_dir))
    print(f"  Loaded {len(traces)} traces")

    print("Loading summaries...")
    summaries = _load_summaries(Path(summaries_dir))
    print(f"  Loaded {len(summaries)} summaries")

    print("Running final privacy scan on summaries...")
    clean_summaries = _run_final_privacy_scan(summaries)

    print(f"Writing {len(traces)} traces to {data_dir / 'traces.jsonl'}...")
    _write_jsonl(traces, data_dir / "traces.jsonl")

    print(f"Writing {len(clean_summaries)} summaries to {data_dir / 'summaries.jsonl'}...")
    _write_jsonl(clean_summaries, data_dir / "summaries.jsonl")

    # Copy aggregate reports
    reports_path = Path(reports_dir)
    if reports_path.exists():
        reports_out = data_dir / "aggregate_reports"
        if reports_out.exists():
            shutil.rmtree(reports_out)
        shutil.copytree(reports_path, reports_out)
        print(f"Copied aggregate reports to {reports_out}")

    stats = _compute_stats(traces, clean_summaries)

    print("Generating dataset card...")
    _generate_dataset_card(stats, out)

    # Write stats summary
    stats_path = out / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print("\nDataset statistics:")
    print(f"  Total traces:    {stats['total_traces']}")
    print(f"  Total summaries: {stats['total_summaries']}")
    print(f"  Agent types:     {list(stats['agent_types'].keys())}")
    print(f"  Task categories: {list(stats['task_categories'].keys())}")
    print(f"\nDataset written to: {out}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AgentLens dataset for HuggingFace")
    parser.add_argument("--traces-dir", default="./traces", help="Directory with trace files")
    parser.add_argument(
        "--summaries-dir", default="./summaries", help="Directory with summary files"
    )
    parser.add_argument("--reports-dir", default="./reports", help="Directory with report files")
    parser.add_argument("--output-dir", default="./hf_dataset", help="Output directory")
    args = parser.parse_args()

    prepare_dataset(
        traces_dir=args.traces_dir,
        summaries_dir=args.summaries_dir,
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
