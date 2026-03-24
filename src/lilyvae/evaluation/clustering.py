"""Clustering evaluation of latent space."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


def evaluate_clustering(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_components: int | None = None,
) -> dict[str, float]:
    """Fit GMM on embeddings and compute ARI against true lithology labels.

    Args:
        embeddings: (N, latent_dim) tensor of embeddings
        labels: (N,) tensor of integer lithology labels
        n_components: Number of GMM components. Defaults to 10 (empirically best).
    """
    z = embeddings.numpy()
    y = labels.numpy()

    if n_components is None:
        n_components = 10

    # Cap components at embedding dimensionality to avoid singular covariance
    n_components = min(n_components, z.shape[1])

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=3, reg_covar=1e-4)
        cluster_labels = gmm.fit_predict(z)
        ari = adjusted_rand_score(y, cluster_labels)
    except ValueError:
        ari = float("nan")

    return {
        "ari": float(ari),
        "n_components": n_components,
    }
