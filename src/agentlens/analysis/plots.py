"""All matplotlib visualizations for agent oversight analysis."""

from __future__ import annotations

from pathlib import Path

from agentlens.analysis.models import (
    AnalysisResults,
    AutonomyAnalysis,
    EscalationAnalysis,
    FailureAnalysis,
    OversightGapAnalysis,
    ToolUsageAnalysis,
)

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#9E9E9E"]
AUTONOMY_LEVELS = ["fully_autonomous", "auto_with_audit", "human_confirmed", "human_driven"]


def _check_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _setup_style():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")


# --- Autonomy Plots (3) ---


def plot_autonomy_by_agent(result: AutonomyAnalysis, output_path: str | Path) -> None:
    """Stacked bar: autonomy levels per agent type."""
    if not _check_matplotlib() or not result.by_agent:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    agents = list(result.by_agent.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = [0.0] * len(agents)
    for i, level in enumerate(AUTONOMY_LEVELS):
        values = [result.by_agent[a].get(level, 0.0) for a in agents]
        ax.bar(agents, values, bottom=bottom, label=level, color=COLORS[i % len(COLORS)])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Agent Type", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title("Autonomy Distribution by Agent Type", fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_autonomy_histogram(result: AutonomyAnalysis, output_path: str | Path) -> None:
    """Histogram: per-session autonomous_action_ratio with mean/median lines."""
    if not _check_matplotlib() or not result.autonomy_ratio_histogram:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(result.autonomy_ratio_histogram, bins=20, color=COLORS[0], alpha=0.7, edgecolor="white")
    ax.axvline(result.mean, color=COLORS[1], linestyle="--", linewidth=2, label=f"Mean={result.mean:.3f}")
    ax.axvline(result.median, color=COLORS[2], linestyle="-.", linewidth=2, label=f"Median={result.median:.3f}")

    ax.set_xlabel("Fully Autonomous Ratio", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Autonomous Action Ratios", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_autonomy_heatmap(result: AutonomyAnalysis, output_path: str | Path) -> None:
    """Heatmap: autonomy level x task category, color = mean fraction."""
    if not _check_matplotlib() or not result.by_task_category:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    tasks = sorted(result.by_task_category.keys())
    levels = AUTONOMY_LEVELS

    data = []
    for task in tasks:
        row = [result.by_task_category[task].get(level, 0.0) for level in levels]
        data.append(row)
    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=10)

    # Annotate cells
    for i in range(len(tasks)):
        for j in range(len(levels)):
            val = data_arr[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Mean Fraction")
    ax.set_title("Autonomy Level by Task Category", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Failure Plots (3) ---


def plot_failure_types(result: FailureAnalysis, output_path: str | Path) -> None:
    """Bar chart: failure types ranked by frequency, color-coded by agent."""
    if not _check_matplotlib() or not result.failure_type_counts:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if result.failure_type_by_agent:
        agents = sorted(result.failure_type_by_agent.keys())
        types = sorted(result.failure_type_counts.keys(),
                       key=lambda x: result.failure_type_counts[x], reverse=True)
        x_pos = range(len(types))
        width = 0.8 / max(len(agents), 1)

        for i, agent in enumerate(agents):
            agent_counts = result.failure_type_by_agent.get(agent, {})
            values = [agent_counts.get(ft, 0) for ft in types]
            offset = (i - len(agents) / 2 + 0.5) * width
            ax.bar([p + offset for p in x_pos], values, width,
                   label=agent, color=COLORS[i % len(COLORS)])

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.legend(fontsize=10)
    else:
        types = sorted(result.failure_type_counts.keys(),
                       key=lambda x: result.failure_type_counts[x], reverse=True)
        counts = [result.failure_type_counts[ft] for ft in types]
        ax.bar(types, counts, color=COLORS[0])
        ax.set_xticklabels(types, rotation=45, ha="right")

    ax.set_xlabel("Failure Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Failure Types by Frequency", fontsize=14)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_failure_by_autonomy(result: FailureAnalysis, output_path: str | Path) -> None:
    """Grouped bar: failure rate by autonomy level and agent type."""
    if not _check_matplotlib() or not result.failure_rate_by_autonomy_level:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted(result.failure_rate_by_autonomy_level.keys())
    rates = [result.failure_rate_by_autonomy_level[l] for l in levels]

    ax.bar(levels, rates, color=[COLORS[i % len(COLORS)] for i in range(len(levels))])
    ax.set_xlabel("Dominant Autonomy Level", fontsize=12)
    ax.set_ylabel("Failure Rate", fontsize=12)
    ax.set_title("Failure Rate by Autonomy Level", fontsize=14)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_graceful_vs_silent(result: FailureAnalysis, output_path: str | Path) -> None:
    """Donut chart: graceful vs silent failures."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [result.graceful_failure_rate, result.silent_failure_rate]
    labels = [
        f"Graceful ({result.graceful_failure_rate:.0%})",
        f"Silent ({result.silent_failure_rate:.0%})",
    ]
    colors_used = [COLORS[2], COLORS[1]]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_used, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75, textprops={"fontsize": 12},
    )
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_patch(centre_circle)
    ax.set_title("Graceful vs Silent Failures", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Tool Usage Plots (4) ---


def plot_tool_frequency(result: ToolUsageAnalysis, output_path: str | Path) -> None:
    """Horizontal bar: tool frequency."""
    if not _check_matplotlib() or not result.tool_frequency:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    tools = list(result.tool_frequency.keys())[:20]
    counts = [result.tool_frequency[t] for t in tools]

    # Color-code by agent if possible
    ax.barh(tools[::-1], counts[::-1], color=COLORS[0])
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Tool", fontsize=12)
    ax.set_title("Tool Usage Frequency", fontsize=14)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tool_bigrams(result: ToolUsageAnalysis, output_path: str | Path) -> None:
    """Horizontal bar: top 10 bigrams."""
    if not _check_matplotlib() or not result.common_bigrams:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    bigrams = result.common_bigrams[:10]
    labels = [bg[0] for bg in bigrams]
    counts = [bg[1] for bg in bigrams]

    ax.barh(labels[::-1], counts[::-1], color=COLORS[3])
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Action Sequence", fontsize=12)
    ax.set_title("Top 10 Action Bigrams", fontsize=14)
    ax.tick_params(labelsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tool_scatter(result: ToolUsageAnalysis, output_path: str | Path) -> None:
    """Scatter: tool frequency (x) vs success rate (y)."""
    if not _check_matplotlib() or not result.tool_frequency:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    tools = list(result.tool_frequency.keys())
    freqs = [result.tool_frequency[t] for t in tools]
    rates = [result.tool_success_rates.get(t, 0.0) for t in tools]

    ax.scatter(freqs, rates, s=100, color=COLORS[0], alpha=0.7, edgecolors="white", linewidth=1)

    for i, tool in enumerate(tools):
        ax.annotate(tool, (freqs[i], rates[i]), fontsize=9, ha="left", va="bottom")

    # Highlight "high freq, low success" quadrant
    if freqs:
        mid_freq = max(freqs) / 2
        ax.axhline(0.8, color=COLORS[1], linestyle="--", alpha=0.5, label="80% success threshold")
        ax.axvline(mid_freq, color=COLORS[5], linestyle="--", alpha=0.5)
        ax.fill_between(
            [mid_freq, max(freqs) * 1.1], 0, 0.8,
            color=COLORS[1], alpha=0.05, label="High freq, low success",
        )

    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Tool Frequency vs Success Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_economic_tasks(result: ToolUsageAnalysis, output_path: str | Path) -> None:
    """Pie chart: economic task distribution."""
    if not _check_matplotlib() or not result.economic_task_distribution:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 8))
    tasks = sorted(result.economic_task_distribution.keys())
    counts = [result.economic_task_distribution[t] for t in tasks]

    colors_used = [COLORS[i % len(COLORS)] for i in range(len(tasks))]
    ax.pie(counts, labels=tasks, colors=colors_used, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title("Economic Task Distribution", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Escalation Plots (2) ---


def plot_escalation_reasons(result: EscalationAnalysis, output_path: str | Path) -> None:
    """Stacked bar: escalation reasons per agent type."""
    if not _check_matplotlib() or not result.reason_distribution:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if result.reason_by_agent:
        agents = sorted(result.reason_by_agent.keys())
        reasons = sorted(result.reason_distribution.keys())
        x_pos = range(len(agents))

        bottom = [0] * len(agents)
        for i, reason in enumerate(reasons):
            values = [result.reason_by_agent.get(a, {}).get(reason, 0) for a in agents]
            ax.bar(x_pos, values, bottom=bottom, label=reason, color=COLORS[i % len(COLORS)])
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(agents, rotation=45, ha="right")
        ax.legend(fontsize=10)
    else:
        reasons = sorted(result.reason_distribution.keys())
        counts = [result.reason_distribution[r] for r in reasons]
        ax.bar(reasons, counts, color=COLORS[0])
        ax.set_xticklabels(reasons, rotation=45, ha="right")

    ax.set_xlabel("Agent Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Escalation Reasons by Agent Type", fontsize=14)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_escalation_matrix(result: EscalationAnalysis, output_path: str | Path) -> None:
    """2x2 matrix: appropriate escalation / false escalation / appropriate autonomy / missed escalation."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate quadrant values
    false_esc = result.false_escalation_estimate
    missed_esc = result.missed_escalation_estimate
    appropriate_esc = 1.0 - false_esc if false_esc <= 1.0 else 0.0
    appropriate_auto = 1.0 - missed_esc if missed_esc <= 1.0 else 0.0

    matrix = [
        [appropriate_esc, false_esc],
        [missed_esc, appropriate_auto],
    ]
    labels_matrix = [
        [f"Appropriate\nEscalation\n{appropriate_esc:.0%}", f"False\nEscalation\n{false_esc:.0%}"],
        [f"Missed\nEscalation\n{missed_esc:.0%}", f"Appropriate\nAutonomy\n{appropriate_auto:.0%}"],
    ]
    cell_colors = [[COLORS[2], COLORS[3]], [COLORS[1], COLORS[2]]]

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=cell_colors[i][j], alpha=0.3))
            ax.text(j + 0.5, 1.5 - i, labels_matrix[i][j],
                    ha="center", va="center", fontsize=13, fontweight="bold")

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Escalated", "Not Escalated"], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Failed", "Succeeded"], fontsize=12)
    ax.set_title("Escalation Decision Matrix", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Oversight Gap Plots (4) ---


def plot_oversight_gap_histogram(result: OversightGapAnalysis, output_path: str | Path) -> None:
    """Histogram: oversight gap score distribution, color by risk tier."""
    if not _check_matplotlib() or not result.score_histogram:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    scores = result.score_histogram

    # Color by risk tier
    low_scores = [s for s in scores if s < 0.3]
    med_scores = [s for s in scores if 0.3 <= s < 0.7]
    high_scores = [s for s in scores if s >= 0.7]

    bins = np.linspace(0, 1, 21)
    if low_scores:
        ax.hist(low_scores, bins=bins, color=COLORS[2], alpha=0.7, label="Low (<0.3)")
    if med_scores:
        ax.hist(med_scores, bins=bins, color=COLORS[3], alpha=0.7, label="Medium (0.3-0.7)")
    if high_scores:
        ax.hist(high_scores, bins=bins, color=COLORS[1], alpha=0.7, label="High (>=0.7)")

    ax.axvline(result.mean_score, color="black", linestyle="--", linewidth=2,
               label=f"Mean={result.mean_score:.3f}")

    ax.set_xlabel("Oversight Gap Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Oversight Gap Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oversight_gap_by_agent(result: OversightGapAnalysis, output_path: str | Path) -> None:
    """Box plot style: oversight gap by agent type (using bar + error concept)."""
    if not _check_matplotlib() or not result.by_agent:
        return
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    agents = sorted(result.by_agent.keys())
    means = [result.by_agent[a] for a in agents]

    ax.bar(agents, means, color=[COLORS[i % len(COLORS)] for i in range(len(agents))], alpha=0.7)

    # Risk tier lines
    ax.axhline(0.3, color=COLORS[2], linestyle="--", alpha=0.5, label="Low/Medium boundary")
    ax.axhline(0.7, color=COLORS[1], linestyle="--", alpha=0.5, label="Medium/High boundary")

    ax.set_xlabel("Agent Type", fontsize=12)
    ax.set_ylabel("Mean Oversight Gap Score", fontsize=12)
    ax.set_title("Oversight Gap by Agent Type", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gap_vs_failure(result: OversightGapAnalysis, output_path: str | Path) -> None:
    """Scatter: gap (x) vs failure rate (y) with trend line and R-squared."""
    if not _check_matplotlib() or not result.score_histogram:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # We only have aggregate data; use score_histogram for x, need sessions for y
    # For the plot we use the top_risk_sessions if available
    if result.top_risk_sessions:
        scores_x = [s["oversight_gap_score"] for s in result.top_risk_sessions]
        # Use consequential as a proxy
        counts_y = [s.get("consequential_action_count", 0) for s in result.top_risk_sessions]
        ax.scatter(scores_x, counts_y, s=100, color=COLORS[0], alpha=0.7,
                   edgecolors="white", linewidth=1)

        if len(scores_x) > 1:
            z = np.polyfit(scores_x, counts_y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(scores_x), max(scores_x), 100)
            ax.plot(x_line, p(x_line), "--", color=COLORS[1], linewidth=2)

        ax.set_ylabel("Consequential Actions", fontsize=12)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)

    r_sq = result.gap_vs_failure ** 2
    ax.set_xlabel("Oversight Gap Score", fontsize=12)
    ax.set_title(f"Gap vs Failure (r={result.gap_vs_failure:.3f}, R²={r_sq:.3f})", fontsize=14)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oversight_gap_heatmap(result: OversightGapAnalysis, output_path: str | Path) -> None:
    """Heatmap: agent type x task category, color = mean gap score."""
    if not _check_matplotlib() or not result.by_agent or not result.by_task_category:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    agents = sorted(result.by_agent.keys())
    tasks = sorted(result.by_task_category.keys())

    # Build a simple matrix using available means (agent mean for all tasks as default)
    data = []
    for agent in agents:
        row = [result.by_agent.get(agent, 0.0)] * len(tasks)
        data.append(row)
    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_arr, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=10)

    for i in range(len(agents)):
        for j in range(len(tasks)):
            val = data_arr[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Mean Oversight Gap Score")
    ax.set_title("Oversight Gap: Agent Type x Task Category", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Master plot function ---


def plot_all(results: AnalysisResults, output_dir: str) -> None:
    """Generate all 16 plots."""
    if not _check_matplotlib():
        print("matplotlib not installed, skipping plot generation")
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Autonomy (3)
    plot_autonomy_by_agent(results.autonomy, plots_dir / "autonomy_by_agent.png")
    plot_autonomy_histogram(results.autonomy, plots_dir / "autonomy_histogram.png")
    plot_autonomy_heatmap(results.autonomy, plots_dir / "autonomy_heatmap.png")

    # Failures (3)
    plot_failure_types(results.failures, plots_dir / "failure_types.png")
    plot_failure_by_autonomy(results.failures, plots_dir / "failure_by_autonomy.png")
    plot_graceful_vs_silent(results.failures, plots_dir / "graceful_vs_silent.png")

    # Tools (4)
    plot_tool_frequency(results.tools, plots_dir / "tool_frequency.png")
    plot_tool_bigrams(results.tools, plots_dir / "tool_bigrams.png")
    plot_tool_scatter(results.tools, plots_dir / "tool_scatter.png")
    plot_economic_tasks(results.tools, plots_dir / "economic_tasks.png")

    # Escalations (2)
    plot_escalation_reasons(results.escalations, plots_dir / "escalation_reasons.png")
    plot_escalation_matrix(results.escalations, plots_dir / "escalation_matrix.png")

    # Oversight Gap (4)
    plot_oversight_gap_histogram(results.oversight_gap, plots_dir / "oversight_gap_histogram.png")
    plot_oversight_gap_by_agent(results.oversight_gap, plots_dir / "oversight_gap_by_agent.png")
    plot_gap_vs_failure(results.oversight_gap, plots_dir / "gap_vs_failure.png")
    plot_oversight_gap_heatmap(results.oversight_gap, plots_dir / "oversight_gap_heatmap.png")
