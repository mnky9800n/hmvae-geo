"""CLI entry point for the canonical bootstrap evaluation.

Runs the unified bootstrap loop defined in
`lilyvae.evaluation.bootstrap.run_bootstrap` and writes a single results
file (`data/results/bootstrap_results.json`) that is the source of truth
for every number in the paper's main results table.

Usage
-----
Full run (every model x 30 folds, with zero-shot):
    uv run lilyvae-bootstrap

Quick sanity check (2 folds, small model set):
    uv run lilyvae-bootstrap --iterations 2 --models pca unsupervised_vae

Skip zero-shot (faster):
    uv run lilyvae-bootstrap --no-zeroshot
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.bootstrap import MODEL_REGISTRY, run_bootstrap


def _make_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


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
            "Subset of model names to run (default: all). "
            f"Available: {', '.join(n for n, _ in MODEL_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--no-zeroshot",
        action="store_true",
        help="Skip the per-fold CatBoost zero-shot evaluation (much faster).",
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
        include_zeroshot=not args.no_zeroshot,
    )

    with open(output_path, "w") as f:
        json.dump(_make_serializable(result), f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Total wall-clock time: {result['elapsed_seconds']/60:.1f} minutes")


if __name__ == "__main__":
    main()
