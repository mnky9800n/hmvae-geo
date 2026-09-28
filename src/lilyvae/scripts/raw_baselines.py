"""CLI for the raw-input baseline comparison.

Answers the question of how hard these tasks actually are, by scoring
the six raw inputs, their principal components, a kNN predictor, a
directly supervised classifier, and the VAE embedding on the same
held-out boreholes. Writes data/results/raw_baselines.json.

Usage
-----
Full run:
    uv run lilyvae-raw-baselines

Quick check without retraining a VAE per fold:
    uv run lilyvae-raw-baselines --iterations 2 --arms raw6 pca
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.raw_baselines import DEFAULT_ARMS, run_raw_baselines
from lilyvae.evaluation.targets import REPORTED_TIERS
from lilyvae.scripts import write_result


def main():
    parser = argparse.ArgumentParser(
        description="Raw inputs vs learned embedding on held-out boreholes."
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--arms", nargs="+", default=list(DEFAULT_ARMS),
        choices=["raw6", "pca", "embedding"],
        help="Feature sets to compare (default: all three).",
    )
    parser.add_argument(
        "--tiers", nargs="+", default=list(REPORTED_TIERS),
        choices=["independent", "same_modality", "duplicate", "procedural"],
    )
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--no-supervised", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR / "raw_baselines.json" if args.output is None else args.output
    )

    result = run_raw_baselines(
        n_iterations=args.iterations,
        test_fraction=args.test_fraction,
        tiers=tuple(args.tiers),
        arms=tuple(args.arms),
        device=args.device,
        include_knn=not args.no_knn,
        include_supervised=not args.no_supervised,
    )

    write_result(output_path, result)

    print(f"\nResults written to {output_path}")
    print(f"Total wall-clock time: {result['elapsed_seconds']/60:.1f} minutes")


if __name__ == "__main__":
    main()
