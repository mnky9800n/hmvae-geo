"""CLI for the zero-shot transfer bootstrap.

Measures zero-shot prediction on boreholes the VAE and the regressor
both never saw, across the same borehole folds as `lilyvae-bootstrap`.
Writes `data/results/zeroshot_bootstrap.json`.

This is the counterpart to the in-sample numbers in
`zeroshot_fulldata.json`: that file says whether a held-out variable is
recoverable from the representation, this one says whether it transfers.
Both are reported; the gap between them is a result about how
non-exchangeable boreholes are.

Usage
-----
Full run (2 VAE variants x 30 folds x curated targets):
    uv run lilyvae-zeroshot-bootstrap

Quick sanity check:
    uv run lilyvae-zeroshot-bootstrap --iterations 2

Include the baselines and ablations:
    uv run lilyvae-zeroshot-bootstrap --models unsupervised_vae pca
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.bootstrap import ALL_TRAINERS
from lilyvae.evaluation.targets import HEADLINE_TIERS, REPORTED_TIERS
from lilyvae.evaluation.zeroshot_bootstrap import run_zeroshot_bootstrap
from lilyvae.scripts import make_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot prediction on held-out boreholes, bootstrapped."
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Trainers to evaluate (default: the two VAE variants). "
            f"Available: {', '.join(n for n, _ in ALL_TRAINERS)}"
        ),
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(REPORTED_TIERS),
        choices=["independent", "same_modality", "duplicate", "procedural"],
        help=(
            "Curation tiers to evaluate. Default is the tiers the paper "
            f"reports ({', '.join(REPORTED_TIERS)}); the headline mean is "
            f"taken over {', '.join(HEADLINE_TIERS)}."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/results/zeroshot_bootstrap.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR / "zeroshot_bootstrap.json"
        if args.output is None
        else args.output
    )

    result = run_zeroshot_bootstrap(
        n_iterations=args.iterations,
        test_fraction=args.test_fraction,
        tiers=tuple(args.tiers),
        device=args.device,
        models=args.models,
    )

    with open(output_path, "w") as f:
        json.dump(make_serializable(result), f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Total wall-clock time: {result['elapsed_seconds']/60:.1f} minutes")


if __name__ == "__main__":
    main()
