"""CLI entry point for single-point full-data evaluation.

Trains each baseline and ablation variant once on all 238K samples and
evaluates on the same data (in-sample, matching the paper's methodology
for non-VAE rows). Writes data/results/single_point_results.json.

This is the counterpart to `lilyvae-bootstrap`: the bootstrap handles the
two full VAE variants (where training-stability CIs are meaningful) and
this script handles everything else (where they are not).

Usage
-----
Full default set (5 baselines + 4 ablation variants):
    uv run lilyvae-single-point

Subset of models:
    uv run lilyvae-single-point --models pca unsupervised_vae

Skip zero-shot (much faster):
    uv run lilyvae-single-point --no-zeroshot
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.bootstrap import ALL_TRAINERS
from lilyvae.evaluation.single_point import SINGLE_POINT_MODELS, run_single_point_eval


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
        description="Single-point full-data evaluation of baselines and ablations."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            f"Subset of models to run. Default: {SINGLE_POINT_MODELS}. "
            f"Available: {[n for n, _ in ALL_TRAINERS]}"
        ),
    )
    parser.add_argument(
        "--no-zeroshot",
        action="store_true",
        help="Skip per-model zero-shot CatBoost evaluation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/results/single_point_results.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR / "single_point_results.json"
        if args.output is None
        else args.output
    )

    result = run_single_point_eval(
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
