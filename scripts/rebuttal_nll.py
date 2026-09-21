"""NLL reconstruction metric for DQpL Q3 (is a likelihood metric more informative than saturated R²?).

Compares the heteroscedastic decoder's Gaussian NLL against a homoscedastic
(fixed per-variable variance) baseline using the SAME predicted means. If the
learned per-sample variances lower NLL, they carry real signal (also relevant
to DQpL W4/Q4). In-sample, consistent with the paper's reconstruction eval.
Writes data/results/rebuttal_nll.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from lilyvae.model.vae import LilyVAE
from lilyvae.config import TRAINING_VARIABLES

PROCESSED = Path("data/processed")
CHECKPOINTS = Path("data/checkpoints")
RESULTS = Path("data/results")
LOG2PI = math.log(2 * math.pi)


def load_vae(name: str) -> LilyVAE:
    vae = LilyVAE()
    sd = torch.load(CHECKPOINTS / f"{name}_vae.pt", map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    vae.load_state_dict(sd)
    vae.eval()
    return vae


def gaussian_nll(x, mu, logvar):
    """Mean Gaussian NLL per element (nats)."""
    return 0.5 * (LOG2PI + logvar + (x - mu) ** 2 / np.exp(logvar))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    features = torch.load(PROCESSED / "train_features.pt")
    x = features.numpy()
    var_names = TRAINING_VARIABLES if len(TRAINING_VARIABLES) == x.shape[1] else \
        [f"var_{i}" for i in range(x.shape[1])]

    out = {}
    for name in ["unsupervised", "semi_supervised"]:
        vae = load_vae(name)
        with torch.no_grad():
            o = vae(features)
        mu = o["x_mu"].numpy()
        logvar = o["x_logvar"].numpy()

        # Heteroscedastic: per-sample, per-variable learned variance
        nll_hetero = gaussian_nll(x, mu, logvar)  # (N, 6)
        # Homoscedastic baseline: one MLE variance per variable, SAME means
        resid2 = (x - mu) ** 2
        sigma2_var = resid2.mean(axis=0, keepdims=True)          # (1, 6)
        logvar_homo = np.log(sigma2_var) * np.ones_like(logvar)
        nll_homo = gaussian_nll(x, mu, logvar_homo)

        per_var_hetero = {v: float(nll_hetero[:, i].mean()) for i, v in enumerate(var_names)}
        out[name] = {
            "nll_heteroscedastic": float(nll_hetero.mean()),
            "nll_homoscedastic_same_means": float(nll_homo.mean()),
            "nll_reduction_from_learned_variance": float(nll_homo.mean() - nll_hetero.mean()),
            "per_variable_nll_heteroscedastic": per_var_hetero,
        }
        print(f"\n=== {name} ===")
        print(f"  heteroscedastic NLL = {out[name]['nll_heteroscedastic']:.4f} nats")
        print(f"  homoscedastic NLL (same means) = {out[name]['nll_homoscedastic_same_means']:.4f} nats")
        print(f"  improvement from learned per-sample variance = "
              f"{out[name]['nll_reduction_from_learned_variance']:.4f} nats")

    (RESULTS / "rebuttal_nll.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'rebuttal_nll.json'}")


if __name__ == "__main__":
    main()
