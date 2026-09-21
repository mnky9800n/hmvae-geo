"""CLI entry point for the depth-confound zero-shot evaluation.

Loads the unsupervised VAE embeddings, the depth metadata, and the
held-out zero-shot targets, then runs the four-way CatBoost comparison
(depth-only, embedding-only, embedding+depth, embedding on
depth-residualised target) for every variable. Writes
data/results/depth_confound.json.

Usage
-----
    uv run lilyvae-depth-confound

The result file mirrors the per_variable structure of
data/results/zeroshot_fulldata.json so plot/table code can join on
variable name.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from lilyvae.config import BIN_SIZE_CM, CHECKPOINT_DIR, PROCESSED_DIR, RESULTS_DIR
from lilyvae.evaluation.depth_confound import evaluate_all_depth_confound
from lilyvae.scripts import make_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Depth-confound check: does the embedding beat depth-only?",
    )
    parser.add_argument(
        "--model",
        default="unsupervised",
        choices=["unsupervised", "semi_supervised"],
        help="Which trained model's embeddings to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/results/depth_confound.json).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        RESULTS_DIR / "depth_confound.json"
        if args.output is None
        else args.output
    )

    embeddings = torch.load(
        CHECKPOINT_DIR / f"{args.model}_embeddings.pt", weights_only=True
    )
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    depth_bins = np.asarray(metadata["depth_bins"])
    depths_m = depth_bins.astype(np.float32) * (BIN_SIZE_CM / 100.0)

    zeroshot_targets = torch.load(
        PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False
    )

    print(f"\nDepth-confound evaluation")
    print(f"  model:     {args.model}")
    print(f"  samples:   {len(depths_m)}")
    print(f"  variables: {len(zeroshot_targets)}")
    print(f"  depth range: {depths_m.min():.1f}--{depths_m.max():.1f} m\n")

    per_variable = evaluate_all_depth_confound(
        embeddings, depths_m, zeroshot_targets
    )

    valid_rows = [
        r for r in per_variable.values()
        if isinstance(r.get("r2_embedding_only"), float)
        and np.isfinite(r["r2_embedding_only"])
    ]
    summary = {}
    for key in (
        "r2_depth_only",
        "r2_embedding_only",
        "r2_embedding_depth",
        "r2_embedding_resid",
    ):
        vals = [
            r[key] for r in valid_rows
            if isinstance(r.get(key), float) and np.isfinite(r[key])
        ]
        summary[key] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "median": float(np.median(vals)) if vals else float("nan"),
            "n": len(vals),
        }

    print(f"\nSummary (mean across {len(valid_rows)} variables):")
    print(f"  depth-only         R^2 = {summary['r2_depth_only']['mean']:.4f}")
    print(f"  embedding-only     R^2 = {summary['r2_embedding_only']['mean']:.4f}")
    print(f"  embedding + depth  R^2 = {summary['r2_embedding_depth']['mean']:.4f}")
    print(f"  embedding on resid R^2 = {summary['r2_embedding_resid']['mean']:.4f}")

    out = {
        "model": args.model,
        "n_samples": int(len(depths_m)),
        "depth_min_m": float(depths_m.min()),
        "depth_max_m": float(depths_m.max()),
        "summary": summary,
        "per_variable": per_variable,
    }

    with open(out_path, "w") as f:
        json.dump(make_serializable(out), f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
