"""UMAP visualizations of the latent space."""

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import umap

from lilyvae.config import PROCESSED_DIR, CHECKPOINT_DIR, PAPER_DIR


def _fit_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_umap_lithology(output_dir: Path | None = None, model_type: str = "unsupervised"):
    """UMAP of latent space colored by lithology (fig_umap_lithology.png)."""
    output_dir = output_dir or PAPER_DIR
    prefix = model_type.replace("-", "_")

    embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True).numpy()
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()

    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    print("  Computing UMAP projection...")
    umap_2d = _fit_umap(embeddings)

    # Color by top lithologies, rest gray
    counts = np.bincount(labels)
    top_n = 10
    top_classes = np.argsort(counts)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot "other" first in gray
    other_mask = ~np.isin(labels, top_classes)
    ax.scatter(umap_2d[other_mask, 0], umap_2d[other_mask, 1], c="lightgray", s=0.5, alpha=0.3, label="Other")

    cmap = plt.cm.tab10
    for i, cls in enumerate(top_classes):
        mask = labels == cls
        name = le.inverse_transform([cls])[0]
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=[cmap(i)], s=1, alpha=0.5, label=name)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5, fontsize=7, loc="best")
    ax.set_title(f"Latent space ({model_type}) colored by lithology")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_umap_lithology.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_umap_lithology.png")


def plot_umap_expeditions(output_dir: Path | None = None, model_type: str = "unsupervised"):
    """UMAP of latent space colored by expedition (fig_umap_expeditions.png)."""
    import pandas as pd

    output_dir = output_dir or PAPER_DIR
    prefix = model_type.replace("-", "_")

    embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True).numpy()
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    borehole_ids = metadata["borehole_ids"]

    # Extract expedition from borehole ID
    expeditions = np.array([bh.split("-")[0] for bh in borehole_ids])

    print("  Computing UMAP projection...")
    umap_2d = _fit_umap(embeddings)

    unique_exp = np.unique(expeditions)
    cmap = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, exp in enumerate(unique_exp):
        mask = expeditions == exp
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=[cmap(i % 20)], s=1, alpha=0.5, label=exp)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5, fontsize=5, loc="best", ncol=2)
    ax.set_title(f"Latent space ({model_type}) colored by expedition")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_umap_expeditions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_umap_expeditions.png")
