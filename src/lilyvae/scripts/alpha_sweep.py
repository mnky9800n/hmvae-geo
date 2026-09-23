"""Sweep the supervision weight and record what it does.

The paper's central claim is that lithology labels add nothing, and the
figure supporting it has until now been drawn from numbers hardcoded in
the plotting function. `data/results/ablations.json` never existed, so
`plot_alpha_sweep` silently fell back to them. That is the one
reproducibility hole under the main result, and this closes it.

Each alpha is trained on every fold of the shared borehole split and
scored on held-out boreholes, so the spread reported is the same kind
of quantity as everywhere else in the paper. Replication comes from the
folds rather than from reseeding one split, because a number that moves
between boreholes is the honest measure of how much a difference
between alphas is worth.

    uv run lilyvae-alpha-sweep --iterations 5
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from lilyvae.config import (
    ALPHA_END,
    HIDDEN_DIMS,
    LATENT_DIM,
    LEARNING_RATE,
    MASK_FRACTION,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    TRAINING_VARIABLES,
    WEIGHT_DECAY,
)
from lilyvae.evaluation.ablations import _train_ablation_model
from lilyvae.evaluation.bootstrap import _make_vae_dataloader
from lilyvae.evaluation.classification import evaluate_classification
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.model.vae import LilyVAE
from lilyvae.scripts import make_serializable
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.trainer import get_device
from sklearn.metrics import r2_score

#: Six orders of magnitude, spanning the point where the classification
#: term and the unsupervised term are comparable per sample. Cross-entropy
#: over 139 classes runs 1-5 nats; reconstruction plus KL runs 5-15. So
#: alpha near 1 is where supervision starts to compete rather than act as
#: a tiebreaker, and 1000 is far past any plausible weighting.
ALPHAS = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 500.0, 1000.0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dev = get_device(args.device)
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])

    folds = list(borehole_splits(borehole_ids, args.iterations,
                                 args.test_fraction, seed=SEED))
    print(f"alpha sweep: {len(args.alphas)} values x {len(folds)} folds "
          f"on {dev}")

    entries, start = [], time.time()
    for alpha in args.alphas:
        per_fold = []
        for fold in folds:
            torch.manual_seed(SEED)
            vae = LilyVAE(input_dim=features.shape[1], latent_dim=LATENT_DIM,
                          hidden_dims=list(HIDDEN_DIMS))
            vae = _train_ablation_model(
                dataloader=_make_vae_dataloader(features[fold.train_mask],
                                                labels[fold.train_mask]),
                vae=vae,
                masking=LearnableMasking(features.shape[1], MASK_FRACTION),
                decoder_type="heteroscedastic", beta_schedule_type="annealing",
                alpha_weight=alpha, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM,
                device=dev,
            )
            vae.eval()
            test = features[fold.test_mask]
            with torch.no_grad():
                out = vae(test.to(dev))
                recon = out["x_mu"].cpu().numpy()
                z = vae.get_embeddings(test.to(dev)).cpu()
            true = test.numpy()
            per_fold.append({
                "r2_avg": float(np.mean([
                    r2_score(true[:, i], recon[:, i])
                    for i in range(len(TRAINING_VARIABLES))
                ])),
                "auc_macro": float(evaluate_classification(
                    z, labels[fold.test_mask])["auc_macro"]),
                "ari": float(evaluate_clustering(z, labels[fold.test_mask])["ari"]),
            })

        def agg(key: str) -> dict:
            v = np.array([f[key] for f in per_fold if np.isfinite(f[key])])
            return {
                "median": float(np.median(v)), "mean": float(np.mean(v)),
                "iqr_low": float(np.percentile(v, 25)),
                "iqr_high": float(np.percentile(v, 75)),
                "folds": [float(x) for x in v],
            }

        entry = {
            "value": alpha,
            # Flat keys the existing plotting code reads.
            "r2_avg": agg("r2_avg")["median"],
            "auc_macro": agg("auc_macro")["median"],
            "ari": agg("ari")["median"],
            "r2_avg_spread": agg("r2_avg"),
            "auc_macro_spread": agg("auc_macro"),
            "ari_spread": agg("ari"),
            "n_folds": len(per_fold),
        }
        entries.append(entry)
        print(f"  alpha {alpha:>8.2f}  R2 {entry['r2_avg']:.4f}  "
              f"AUC {entry['auc_macro']:.4f}  ARI {entry['ari']:.4f}", flush=True)

    result = {
        "config": {
            "alphas": args.alphas, "n_iterations": args.iterations,
            "test_fraction": args.test_fraction, "default_alpha": ALPHA_END,
            "seed": SEED, "elapsed_seconds": time.time() - start,
        },
        "ablations": {"alpha_weight": entries},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (RESULTS_DIR / "ablations.json")
    with open(out, "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
