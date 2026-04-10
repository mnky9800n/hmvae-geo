"""CLI entry point for the canonical bootstrap evaluation.

Runs the bootstrap loop defined in `lilyvae.evaluation.bootstrap.run_bootstrap`
and writes a single results file (`data/results/bootstrap_results.json`)
that is the source of truth for the bootstrap training-stability metrics
(reconstruction R^2, ARI, AUC) reported in the paper.

Zero-shot R^2 is intentionally NOT computed here. It is a property of a
full-data trained model's embedding space and is measured separately via
`lilyvae-evaluate` on the full-data VAE checkpoints. Running CatBoost per
fold on 18k held-out samples makes CatBoost undersampled and inflates
R^2 for trivial "embeddings" (e.g., PCA as identity), which is not what
the paper is claiming.

Usage
-----
Full run (every model x 30 folds):
    uv run lilyvae-bootstrap

Quick sanity check (2 folds, small model set):
    uv run lilyvae-bootstrap --iterations 2 --models pca unsupervised_vae
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
    )

    with open(output_path, "w") as f:
        json.dump(_make_serializable(result), f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Total wall-clock time: {result['elapsed_seconds']/60:.1f} minutes")


if __name__ == "__main__":
    main()
