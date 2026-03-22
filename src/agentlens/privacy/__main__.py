"""Entry point for `python -m agentlens.privacy`."""

import argparse
import asyncio
import os

from agentlens.privacy.runner import run_full_privacy_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgentLens privacy validation experiments",
    )
    parser.add_argument(
        "--traces-dir",
        default=None,
        help="Directory containing trace files (JSON/JSONL). If omitted, generates synthetic traces.",
    )
    parser.add_argument(
        "--output-dir",
        default="./privacy_results",
        help="Directory to save results (default: ./privacy_results)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock mode (no API calls, deterministic)",
    )
    parser.add_argument(
        "--num-pii-traces",
        type=int,
        default=50,
        help="Number of PII-injected traces to test (default: 50)",
    )
    parser.add_argument(
        "--num-reident-trials",
        type=int,
        default=3,
        help="Trials per batch size for re-identification (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.mock and not api_key:
        parser.error(
            "ANTHROPIC_API_KEY environment variable is required when not using --mock"
        )

    asyncio.run(run_full_privacy_validation(
        traces_dir=args.traces_dir,
        output_dir=args.output_dir,
        use_mock=args.mock,
        num_pii_traces=args.num_pii_traces,
        num_reident_trials=args.num_reident_trials,
        api_key=api_key,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
