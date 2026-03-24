"""Reconstruction quality metrics."""

import torch
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from lilyvae.model.vae import LilyVAE
from lilyvae.config import TRAINING_VARIABLES


def evaluate_reconstruction(
    vae: LilyVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute per-variable and average R² on the dataset."""
    vae.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for features, _, _ in dataloader:
            features = features.to(device)
            out = vae(features)
            all_true.append(features.cpu().numpy())
            all_pred.append(out["x_mu"].cpu().numpy())

    true = np.concatenate(all_true, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    var_names = TRAINING_VARIABLES if len(TRAINING_VARIABLES) == true.shape[1] else [
        f"var_{i}" for i in range(true.shape[1])
    ]

    results = {}
    for i, name in enumerate(var_names):
        results[f"r2_{name}"] = float(r2_score(true[:, i], pred[:, i]))
    results["r2_avg"] = float(np.mean([v for k, v in results.items() if k.startswith("r2_")]))

    return results
