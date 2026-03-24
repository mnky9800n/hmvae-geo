"""CLI entry point for zero-shot evaluation and Monte Carlo exploration."""

import argparse
import json
import torch
import numpy as np

from lilyvae.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_TEST_FRACTION,
)
from lilyvae.evaluation.bootstrap import bootstrap_zeroshot
from lilyvae.evaluation.monte_carlo import monte_carlo_explore


def main():
    parser = argparse.ArgumentParser(description="Zero-shot prediction evaluation")
    parser.add_argument("--model", choices=["unsupervised", "semi-supervised"], default="unsupervised")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo exploration")
    parser.add_argument("--mc-trials", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.model.replace("-", "_")

    embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    borehole_ids = metadata["borehole_ids"]

    if args.monte_carlo:
        print(f"Running Monte Carlo exploration ({args.mc_trials} trials)...")
        mc_results = monte_carlo_explore(
            embeddings, zeroshot_targets, borehole_ids, n_trials=args.mc_trials,
        )
        mc_path = RESULTS_DIR / f"{prefix}_monte_carlo.csv"
        mc_results.to_csv(mc_path, index=False)
        print(f"Monte Carlo results saved to {mc_path}")

        # Summary: best config per target
        print("\nBest configurations per target variable:")
        for target in mc_results["target"].unique():
            subset = mc_results[mc_results["target"] == target]
            best = subset.loc[subset["r2"].idxmax()]
            print(f"  {target}: R²={best['r2']:.4f} (dims={best['n_dims']}, depth={best['catboost_depth']}, lr={best['catboost_lr']})")
    else:
        print(f"Running bootstrap zero-shot evaluation ({args.bootstrap_iterations} iterations)...")
        results = {}
        for name, target in zeroshot_targets.items():
            print(f"\n  {name} ({target['n_valid']} valid samples)...")
            try:
                result = bootstrap_zeroshot(
                    embeddings,
                    target["values"],
                    target["valid_mask"],
                    borehole_ids,
                    n_iterations=args.bootstrap_iterations,
                )
                results[name] = result
            except Exception as e:
                print(f"    ERROR: {e}")
                results[name] = {"r2_mean": float("nan"), "r2_ci_low": float("nan"), "r2_ci_high": float("nan"), "error": str(e)}
            print(f"    R² = {result['r2_mean']:.4f} [{result['r2_ci_low']:.4f}, {result['r2_ci_high']:.4f}]")

        results_path = RESULTS_DIR / f"{prefix}_bootstrap_zeroshot.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
