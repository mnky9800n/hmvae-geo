"""CLI entry point for bootstrap VAE training evaluation."""

import argparse
import json
import numpy as np
import torch

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.bootstrap_vae import bootstrap_vae_training


def main():
    parser = argparse.ArgumentParser(description="Bootstrap VAE training across borehole splits")
    parser.add_argument("--model", choices=["unsupervised", "semi-supervised", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = ["unsupervised", "semi-supervised"] if args.model == "both" else [args.model]

    for model_type in models:
        result = bootstrap_vae_training(
            model_type=model_type,
            n_iterations=args.iterations,
            test_fraction=args.test_fraction,
            device=args.device,
        )

        prefix = model_type.replace("-", "_")

        # Save summary
        summary_path = RESULTS_DIR / f"{prefix}_bootstrap_vae.json"
        with open(summary_path, "w") as f:
            json.dump({
                "summary": result["summary"],
                "per_iteration": result["per_iteration"],
                "n_iterations": result["n_iterations"],
                "test_fraction": result["test_fraction"],
            }, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        # Save aggregated predictions for plotting
        torch.save({
            "true": torch.from_numpy(result["aggregated_true"]),
            "pred": torch.from_numpy(result["aggregated_pred"]),
            "indices": torch.from_numpy(result["aggregated_indices"]),
        }, RESULTS_DIR / f"{prefix}_bootstrap_predictions.pt")
        print(f"Aggregated predictions saved for plotting")


if __name__ == "__main__":
    main()
