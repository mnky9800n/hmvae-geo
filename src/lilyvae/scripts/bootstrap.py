"""CLI entry point for the VAE training-stability bootstrap.

Runs the bootstrap loop defined in `lilyvae.evaluation.bootstrap.run_bootstrap`
for the two full VAE variants (unsupervised + semi-supervised) across 30
borehole-level splits. Writes `data/results/bootstrap_results.json`,
which is the source of truth for the VAE rows' CIs in the paper's main
results table.

By default only the VAE variants are bootstrapped. Pass --all-models to
bootstrap the baselines and ablations on the same folds; the paper's main
table needs that, because reporting bootstrapped VAEs beside single-run
in-sample baselines compares protocols rather than architectures.

Zero-shot R^2 is not computed here. It is a property of the full-data
trained VAE's embedding space and is measured separately via the
pipeline's zero-shot step.

Usage
-----
Full run (2 VAE variants x 30 folds, matches the paper):
    uv run lilyvae-bootstrap

Every row of the main table under one protocol (9 models x 30 folds):
    uv run lilyvae-bootstrap --all-models

Quick sanity check (2 folds):
    uv run lilyvae-bootstrap --iterations 2
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.bootstrap import ALL_TRAINERS, run_bootstrap
from lilyvae.scripts import write_result


def main():
    parser = argparse.ArgumentParser(
        description="Canonical bootstrap evaluation across all models."
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Subset of model names to run (default: the two VAE variants). "
            f"Available: {', '.join(n for n, _ in ALL_TRAINERS)}"
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Bootstrap every baseline and ablation as well as the two VAEs, "
            "so every row of the paper's main table is measured under one "
            "protocol instead of mixing bootstrap and single-run in-sample."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/results/bootstrap_results.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR / "bootstrap_results.json"
        if args.output is None
        else args.output
    )

    result = run_bootstrap(
        n_iterations=args.iterations,
        test_fraction=args.test_fraction,
        device=args.device,
        models=args.models,
        all_models=args.all_models,
    )

    write_result(output_path, result)

    print(f"\nResults written to {output_path}")
    print(f"Total wall-clock time: {result['elapsed_seconds']/60:.1f} minutes")


if __name__ == "__main__":
    main()
