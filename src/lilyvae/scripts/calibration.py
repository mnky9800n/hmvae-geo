"""CLI for the decoder calibration analysis.

Scores the heteroscedastic decoder's predicted variances as
uncertainties, and optionally retrains with wider log-variance bounds to
test whether the clamp floor is what makes them uninformative.
Writes data/results/calibration.json.

Usage
-----
Score the shipped checkpoints:
    uv run lilyvae-calibration

Also retrain at three clamp settings (needs a GPU; ~3 full trainings):
    uv run lilyvae-calibration --clamp-sweep
"""

from __future__ import annotations

import argparse
import json

import torch

from lilyvae.config import CHECKPOINT_DIR, PROCESSED_DIR, RESULTS_DIR
from lilyvae.evaluation.calibration import calibration_from_model, clamp_sweep
from lilyvae.model.vae import LilyVAE
from lilyvae.scripts import write_result


def _load_vae(name: str) -> LilyVAE:
    vae = LilyVAE()
    state = torch.load(CHECKPOINT_DIR / f"{name}_vae.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser(
        description="Calibration of the heteroscedastic decoder's variances."
    )
    parser.add_argument(
        "--models", nargs="+", default=["unsupervised", "semi_supervised"]
    )
    parser.add_argument(
        "--clamp-sweep", action="store_true",
        help="Retrain at several log-variance clamp settings (slow).",
    )
    parser.add_argument(
        "--sweep-epochs", type=int, default=None,
        help="Epochs per clamp setting (default: the configured NUM_EPOCHS).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR / "calibration.json" if args.output is None else args.output
    )

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    results: dict = {"checkpoints": {}}

    for name in args.models:
        print(f"\n=== {name} ===")
        res = calibration_from_model(_load_vae(name), features)
        results["checkpoints"][name] = res
        print(f"  NLL heteroscedastic          {res['nll_heteroscedastic']:.4f} nats")
        print(f"  NLL homoscedastic, same means {res['nll_homoscedastic_same_means']:.4f} nats")
        print(f"  mean standardised residual^2  {res['mean_standardised_residual_sq']:.4f}"
              "   (1.0 when calibrated)")
        print(f"  fraction at clamp floor       {res['fraction_at_clamp_floor']:.4f}")
        print(f"  spearman(variance, |error|)   {res['spearman_variance_vs_abs_error']:.4f}")
        for level, cov in res["coverage"].items():
            print(f"    nominal {cov['nominal']:.2f} -> empirical {cov['empirical']:.4f}")

    if args.clamp_sweep:
        labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
        print("\n=== clamp sweep ===")
        results["clamp_sweep"] = clamp_sweep(
            features, labels, num_epochs=args.sweep_epochs, device=args.device
        )

    write_result(output_path, results)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
