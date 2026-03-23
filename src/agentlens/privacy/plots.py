"""Publication-quality matplotlib charts for privacy validation results."""

from __future__ import annotations

from pathlib import Path

from agentlens.privacy.reidentification_test import ReidentificationResult
from agentlens.privacy.utility_tradeoff import UtilityPrivacyReport


def _check_matplotlib():
    try:
        import importlib.util
        return importlib.util.find_spec("matplotlib") is not None
    except ImportError:
        return False


def plot_reidentification_vs_batch_size(
    results: list[ReidentificationResult],
    output_path: str | Path,
) -> None:
    """Line chart: TPR + FPR vs batch size with random baseline."""
    if not _check_matplotlib():
        print("matplotlib not installed, skipping plot generation")
        return

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    batch_sizes = [r.batch_size for r in results]
    tprs = [r.true_positive_rate for r in results]
    fprs = [r.false_positive_rate for r in results]
    baselines = [r.random_baseline_tpr for r in results]

    ax.plot(
        batch_sizes, tprs, "o-",
        color="#2196F3", linewidth=2, markersize=8, label="True Positive Rate",
    )
    ax.plot(
        batch_sizes, fprs, "s-",
        color="#F44336", linewidth=2, markersize=8, label="False Positive Rate",
    )
    ax.plot(batch_sizes, baselines, "--", color="#9E9E9E", linewidth=1.5, label="Random Baseline")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Re-identification Rate vs Batch Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_utility_privacy_tradeoff(
    report: UtilityPrivacyReport,
    output_path: str | Path,
) -> None:
    """Dual-axis chart: utility (blue) and leakage rate (red) vs granularity level."""
    if not _check_matplotlib():
        print("matplotlib not installed, skipping plot generation")
        return

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    levels = report.levels
    utilities = report.mean_utility_scores
    leakages = report.mean_leakage_rates

    # Left axis: utility
    color_util = "#2196F3"
    ax1.set_xlabel("Granularity Level", fontsize=12)
    ax1.set_ylabel("Mean Utility Score", fontsize=12, color=color_util)
    ax1.plot(levels, utilities, "o-", color=color_util, linewidth=2, markersize=8, label="Utility")
    ax1.tick_params(axis="y", labelcolor=color_util, labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)

    # Right axis: leakage
    ax2 = ax1.twinx()
    color_leak = "#F44336"
    ax2.set_ylabel("Leakage Rate", fontsize=12, color=color_leak)
    ax2.plot(levels, leakages, "s-", color=color_leak, linewidth=2, markersize=8, label="Leakage")
    ax2.tick_params(axis="y", labelcolor=color_leak, labelsize=11)

    # Star marker on recommended level
    rec_idx = levels.index(report.recommended_level) if report.recommended_level in levels else 0
    ax1.plot(
        levels[rec_idx], utilities[rec_idx], "*",
        color="#4CAF50", markersize=20, zorder=5,
        label=f"Recommended (L{report.recommended_level})",
    )

    ax1.set_title("Utility-Privacy Trade-off", fontsize=14)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="center left")

    ax1.set_xticks(levels)
    ax2.set_ylim(-0.05, 1.05)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
