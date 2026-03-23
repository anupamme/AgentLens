"""Markdown report generator for analysis results."""

from __future__ import annotations

from pathlib import Path

from agentlens.analysis.models import AnalysisResults


def generate_analysis_report(results: AnalysisResults, output_path: str) -> None:
    """Generate a deterministic markdown report from analysis results."""
    a = results.autonomy
    f = results.failures
    t = results.tools
    e = results.escalations
    o = results.oversight_gap
    meta = results.metadata

    sections: list[str] = []

    # 1. Executive Summary
    sections.append("# Agent Oversight Analysis Report\n")
    sections.append("## 1. Executive Summary\n")
    n = meta.get("session_count", 0)
    sections.append(
        f"This report analyzes **{n} agent sessions** across "
        f"**{len(meta.get('agent_types', []))} agent type(s)** and "
        f"**{len(meta.get('task_categories', []))} task category/categories**.\n"
    )
    sections.append(
        f"- Mean oversight gap score: **{o.mean_score:.3f}**\n"
        f"- Overall failure rate: **{f.overall_failure_rate:.2%}**\n"
        f"- Overall escalation rate: **{e.overall_escalation_rate:.2%}**\n"
        f"- High autonomy sessions: **{a.high_autonomy_session_count}** "
        f"({a.high_autonomy_fraction:.1%})\n"
        f"- High risk sessions: **{o.high_risk_count}**\n"
    )

    # 2. Dataset Overview
    sections.append("## 2. Dataset Overview\n")
    sections.append(f"- **Sessions**: {n}\n")
    sections.append(
        f"- **Agent types**: {', '.join(meta.get('agent_types', []))}\n"
    )
    sections.append(
        f"- **Task categories**: {', '.join(meta.get('task_categories', []))}\n"
    )
    generated = meta.get("generated_at", "N/A")
    sections.append(f"- **Generated at**: {generated}\n")

    # 3. Autonomy Profiling
    sections.append("## 3. Autonomy Profiling\n")
    sections.append("### Overall Autonomy Distribution\n")
    for level, val in sorted(a.overall_distribution.items()):
        sections.append(f"- {level}: {val:.2%}\n")
    sections.append(
        f"\nMean fully-autonomous ratio: **{a.mean:.3f}** "
        f"(median: {a.median:.3f}, std: {a.std:.3f})\n"
    )
    if a.by_agent:
        sections.append("\n### Autonomy by Agent\n")
        for agent, dist in sorted(a.by_agent.items()):
            fa = dist.get("fully_autonomous", 0.0)
            sections.append(f"- **{agent}**: fully_autonomous={fa:.2%}\n")
    sections.append("\n![Autonomy by Agent](plots/autonomy_by_agent.png)\n")
    sections.append("![Autonomy Histogram](plots/autonomy_histogram.png)\n")
    sections.append("![Autonomy Heatmap](plots/autonomy_heatmap.png)\n")

    # 4. Failure Taxonomy
    sections.append("## 4. Failure Taxonomy\n")
    sections.append(f"- Overall failure rate: **{f.overall_failure_rate:.2%}**\n")
    sections.append(
        f"- Graceful failure rate: **{f.graceful_failure_rate:.2%}**\n"
        f"- Silent failure rate: **{f.silent_failure_rate:.2%}**\n"
    )
    if f.failure_type_counts:
        sections.append("\n### Failure Types\n")
        for ft, count in sorted(f.failure_type_counts.items(), key=lambda x: -x[1]):
            sections.append(f"- {ft}: {count}\n")
    if f.failure_rate_by_autonomy_level:
        sections.append("\n### Failure Rate by Autonomy Level\n")
        for level, rate in sorted(f.failure_rate_by_autonomy_level.items()):
            sections.append(f"- {level}: {rate:.2%}\n")
    sections.append("\n![Failure Types](plots/failure_types.png)\n")
    sections.append("![Failure by Autonomy](plots/failure_by_autonomy.png)\n")
    sections.append("![Graceful vs Silent](plots/graceful_vs_silent.png)\n")

    # 5. Tool Usage Patterns
    sections.append("## 5. Tool Usage Patterns\n")
    sections.append(
        f"- Average unique tools per session: **{t.avg_unique_tools_per_session:.1f}**\n"
    )
    if t.tool_frequency:
        sections.append("\n### Top Tools\n")
        for tool, count in list(t.tool_frequency.items())[:10]:
            rate = t.tool_success_rates.get(tool, 0.0)
            sections.append(f"- {tool}: {count} uses (success: {rate:.0%})\n")
    if t.problematic_tools:
        sections.append(
            f"\n**Problematic tools** (success < 80%): {', '.join(t.problematic_tools)}\n"
        )
    if t.common_bigrams:
        sections.append("\n### Common Action Sequences (Bigrams)\n")
        for bg, count in t.common_bigrams[:10]:
            sections.append(f"- {bg}: {count}\n")
    sections.append("\n![Tool Frequency](plots/tool_frequency.png)\n")
    sections.append("![Top Bigrams](plots/tool_bigrams.png)\n")
    sections.append("![Tool Scatter](plots/tool_scatter.png)\n")
    sections.append("![Economic Tasks](plots/economic_tasks.png)\n")

    # 6. Escalation Analysis
    sections.append("## 6. Escalation Analysis\n")
    sections.append(f"- Overall escalation rate: **{e.overall_escalation_rate:.2%}**\n")
    sections.append(
        f"- False escalation estimate: **{e.false_escalation_estimate:.2%}**\n"
        f"- Missed escalation estimate: **{e.missed_escalation_estimate:.2%}**\n"
    )
    if e.mean_actions_before_first_escalation > 0:
        sections.append(
            f"- Mean actions before first escalation: "
            f"**{e.mean_actions_before_first_escalation:.1f}**\n"
        )
    if e.reason_distribution:
        sections.append("\n### Escalation Reasons\n")
        for reason, count in sorted(e.reason_distribution.items(), key=lambda x: -x[1]):
            sections.append(f"- {reason}: {count}\n")
    sections.append("\n![Escalation Reasons](plots/escalation_reasons.png)\n")
    sections.append("![Escalation Matrix](plots/escalation_matrix.png)\n")

    # 7. Oversight Gap Analysis
    sections.append("## 7. Oversight Gap Analysis\n")
    sections.append(
        f"- Mean score: **{o.mean_score:.3f}** "
        f"(median: {o.median_score:.3f}, std: {o.std_score:.3f})\n"
    )
    sections.append(
        f"- Risk tiers: Low={o.low_risk_count}, "
        f"Medium={o.medium_risk_count}, High={o.high_risk_count}\n"
    )
    sections.append(
        f"\n### Correlations\n"
        f"- Gap vs failure rate: r={o.gap_vs_failure:.3f}\n"
        f"- Gap vs duration: r={o.gap_vs_duration:.3f}\n"
        f"- Gap vs action count: r={o.gap_vs_action_count:.3f}\n"
    )
    if o.top_risk_sessions:
        sections.append("\n### Top Risk Sessions\n")
        sections.append("| Session | Agent | Score | Consequential |\n")
        sections.append("|---------|-------|-------|---------------|\n")
        for s in o.top_risk_sessions[:5]:
            sections.append(
                f"| {s['session_id']} | {s['agent_type']} | "
                f"{s['oversight_gap_score']:.3f} | {s['consequential_action_count']} |\n"
            )
    sections.append("\n![Oversight Gap Distribution](plots/oversight_gap_histogram.png)\n")
    sections.append("![Oversight Gap by Agent](plots/oversight_gap_by_agent.png)\n")
    sections.append("![Gap vs Failure](plots/gap_vs_failure.png)\n")
    sections.append("![Gap Heatmap](plots/oversight_gap_heatmap.png)\n")

    # 8. Key Findings
    sections.append("## 8. Key Findings\n")
    findings = _extract_findings(results)
    for finding in findings:
        sections.append(f"- {finding}\n")

    # 9. Implications for Agent Safety
    sections.append("\n## 9. Implications for Agent Safety\n")
    sections.append(
        "- Agents with high autonomy and high failure rates require additional guardrails.\n"
        "- Missed escalations indicate gaps in the agent's self-assessment capabilities.\n"
        "- High oversight gap scores correlate with unsupervised consequential actions.\n"
        "- Tool success rates below 80% warrant investigation and potential deprecation.\n"
    )

    # 10. Limitations
    sections.append("\n## 10. Limitations\n")
    sections.append(
        "- Escalation timing is approximated since exact action-level timestamps "
        "are not available in session summaries.\n"
        "- Tool success rates are distributed evenly across tools per session, "
        "which may not reflect actual per-tool performance.\n"
        "- Autonomy-failure correlation uses dominant autonomy level bucketing, "
        "which loses granularity for mixed-autonomy sessions.\n"
        "- Economic task mapping uses a static dictionary and may not cover all tools.\n"
    )

    # Write report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(sections))


def _extract_findings(results: AnalysisResults) -> list[str]:
    """Extract key findings from metrics."""
    findings: list[str] = []
    a = results.autonomy
    f = results.failures
    e = results.escalations
    o = results.oversight_gap

    if a.high_autonomy_fraction > 0.5:
        findings.append(
            f"Over half ({a.high_autonomy_fraction:.0%}) of sessions operate "
            f"with high autonomy (>80% fully autonomous)."
        )

    if f.overall_failure_rate > 0.1:
        findings.append(
            f"Overall failure rate ({f.overall_failure_rate:.1%}) exceeds 10% threshold."
        )

    if f.silent_failure_rate > 0.3:
        findings.append(
            f"Silent failures ({f.silent_failure_rate:.0%}) represent a significant "
            f"portion of failure handling."
        )

    if e.missed_escalation_estimate > 0.1:
        findings.append(
            f"Missed escalation rate ({e.missed_escalation_estimate:.1%}) suggests "
            f"agents are not requesting help when needed."
        )

    if e.false_escalation_estimate > 0.5:
        findings.append(
            f"High false escalation rate ({e.false_escalation_estimate:.0%}) "
            f"suggests over-cautious escalation behavior."
        )

    if o.high_risk_count > 0:
        findings.append(
            f"{o.high_risk_count} session(s) scored in the high-risk oversight gap tier."
        )

    if abs(o.gap_vs_failure) > 0.5:
        findings.append(
            f"Strong correlation (r={o.gap_vs_failure:.2f}) between oversight gap "
            f"and failure rate."
        )

    if not findings:
        findings.append("No significant findings detected in this analysis.")

    return findings
