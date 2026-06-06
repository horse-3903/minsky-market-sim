"""Entry point: run baseline experiment and generate outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim_minsky_market.experiments.baseline_stability import run as run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Minsky Market Simulation")
    parser.add_argument(
        "--experiment",
        choices=["baseline"],
        default="baseline",
        help="Which experiment to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results", help="Output directory root")
    args = parser.parse_args()

    if args.experiment == "baseline":
        stats = run_baseline(seed=args.seed, output_dir=f"{args.output}/baseline")
        print("\nSummary statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
