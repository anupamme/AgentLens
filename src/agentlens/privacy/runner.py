"""Orchestrates all 3 privacy experiments, saves results + markdown report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agentlens.aggregation.aggregator import BaseAggregator, MockAggregator
from agentlens.aggregation.summarizer import BaseSummarizer, MockSummarizer
from agentlens.privacy.leakage_test import PIILeakageReport, PIILeakageTest
from agentlens.privacy.pii_generator import PIIGenerator
from agentlens.privacy.reidentification_test import (
    MockAdversary,
    ReidentificationResult,
    ReidentificationTest,
)
from agentlens.privacy.trace_factory import make_diverse_traces
from agentlens.privacy.utility_tradeoff import UtilityPrivacyAnalysis, UtilityPrivacyReport


async def run_full_privacy_validation(
    traces_dir: str | None = None,
    output_dir: str = "./privacy_results",
    use_mock: bool = True,
    num_pii_traces: int = 50,
    num_reident_trials: int = 3,
    api_key: str | None = None,
    seed: int = 42,
) -> dict:
    """Run all three privacy validation experiments.

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up components
    if use_mock:
        summarizer: BaseSummarizer = MockSummarizer()
        aggregator: BaseAggregator = MockAggregator()
        adversary = MockAdversary(seed=seed)
    else:
        from agentlens.aggregation.aggregator import SessionAggregator
        from agentlens.aggregation.summarizer import SessionSummarizer
        from agentlens.privacy.reidentification_test import LLMAdversary

        summarizer = SessionSummarizer(api_key=api_key)
        aggregator = SessionAggregator(api_key=api_key)
        adversary = LLMAdversary(api_key=api_key)

    pii_gen = PIIGenerator(seed=seed)

    # Load or generate traces
    if traces_dir:
        traces_path = Path(traces_dir)
        from agentlens.schema.trace import SessionTrace
        traces = []
        for p in traces_path.glob("**/*.json"):
            traces.append(SessionTrace.from_json(p.read_text()))
        for p in traces_path.glob("**/*.jsonl"):
            for line in p.read_text().splitlines():
                if line.strip():
                    traces.append(SessionTrace.from_json(line.strip()))
    else:
        traces = make_diverse_traces(count=max(num_pii_traces, 120), seed=seed)

    # --- Experiment 1: PII Leakage ---
    print("Running PII leakage test...")
    leakage_test = PIILeakageTest(summarizer, aggregator, pii_gen)
    leakage_report = await leakage_test.run(num_traces=num_pii_traces)

    leakage_path = output_path / "pii_leakage_results.json"
    leakage_path.write_text(json.dumps(asdict(leakage_report), indent=2, default=str))

    # --- Experiment 2: Re-identification ---
    print("Running re-identification sweep...")
    reident_test = ReidentificationTest(summarizer, aggregator, adversary)

    # Filter batch sizes that fit available traces
    max_traces = len(traces)
    batch_sizes = [s for s in [5, 10, 20, 50, 100] if s + 20 <= max_traces]

    reident_results = await reident_test.run_batch_size_sweep(
        all_traces=traces,
        batch_sizes=batch_sizes,
        num_decoys=20,
        trials_per_batch=num_reident_trials,
        seed=seed,
    )

    reident_path = output_path / "reidentification_results.json"
    reident_path.write_text(json.dumps(
        [asdict(r) for r in reident_results], indent=2,
    ))

    # --- Experiment 3: Utility-Privacy Trade-off ---
    print("Running utility-privacy analysis...")
    pii_gen_utility = PIIGenerator(seed=seed + 1)
    utility_analysis = UtilityPrivacyAnalysis(pii_gen_utility)
    utility_report = await utility_analysis.run(
        traces=traces[:30],
        granularity_levels=[1, 2, 3, 4, 5],
    )

    utility_path = output_path / "utility_privacy_results.json"
    utility_path.write_text(json.dumps(asdict(utility_report), indent=2))

    # --- Generate Plots ---
    print("Generating plots...")
    try:
        from agentlens.privacy.plots import (
            plot_reidentification_vs_batch_size,
            plot_utility_privacy_tradeoff,
        )
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        if reident_results:
            plot_reidentification_vs_batch_size(
                reident_results, plots_dir / "reidentification_vs_batch_size.png",
            )
        plot_utility_privacy_tradeoff(
            utility_report, plots_dir / "utility_privacy_tradeoff.png",
        )
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    # --- Generate Markdown Report ---
    md_report = _generate_markdown_report(leakage_report, reident_results, utility_report)
    md_path = output_path / "privacy_validation_report.md"
    md_path.write_text(md_report)

    print(f"Results saved to {output_path}")

    return {
        "leakage": asdict(leakage_report),
        "reidentification": [asdict(r) for r in reident_results],
        "utility_privacy": asdict(utility_report),
    }


def _generate_markdown_report(
    leakage: PIILeakageReport,
    reident: list[ReidentificationResult],
    utility: UtilityPrivacyReport,
) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        "# Privacy Validation Report",
        "",
        "## 1. PII Leakage Test",
        "",
        f"- **Traces tested**: {leakage.num_traces_tested}",
        f"- **Stage 1 leakage rate**: {leakage.stage1_leakage_rate:.1%}",
        f"- **Stage 2 leakage rate**: {leakage.stage2_leakage_rate:.1%}",
        f"- **Overall pass**: {'PASS' if leakage.overall_pass else 'FAIL'}",
        "",
    ]

    if leakage.stage1_leaked_types:
        lines.append("### Stage 1 Leaked PII Types")
        lines.append("")
        for pii_type, count in sorted(leakage.stage1_leaked_types.items()):
            lines.append(f"- {pii_type}: {count}")
        lines.append("")

    lines.extend([
        "## 2. Re-identification Attack",
        "",
        "| Batch Size | TPR | FPR | Precision | Recall | F1 | Random Baseline |",
        "|-----------|-----|-----|-----------|--------|----|-----------------| ",
    ])
    for r in reident:
        lines.append(
            f"| {r.batch_size} | {r.true_positive_rate:.3f} | {r.false_positive_rate:.3f} "
            f"| {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.random_baseline_tpr:.3f} |"
        )
    lines.append("")

    lines.extend([
        "## 3. Utility-Privacy Trade-off",
        "",
        "| Level | Utility | Leakage Rate |",
        "|-------|---------|--------------|",
    ])
    for i, level in enumerate(utility.levels):
        marker = " *" if level == utility.recommended_level else ""
        lines.append(
            f"| {level}{marker} | {utility.mean_utility_scores[i]:.3f} "
            f"| {utility.mean_leakage_rates[i]:.3f} |"
        )
    lines.append("")
    lines.append(f"**Recommendation**: {utility.recommendation_rationale}")
    lines.append("")

    return "\n".join(lines)
