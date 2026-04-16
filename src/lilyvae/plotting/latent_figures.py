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


def plot_umap_comparison(output_dir: Path | None = None):
    """Side-by-side UMAP of unsupervised vs semi-supervised latent space (fig_umap_comparison.png)."""
    output_dir = output_dir or PAPER_DIR

    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    counts = np.bincount(labels)
    top_n = 10
    top_classes = np.argsort(counts)[::-1][:top_n]
    other_mask = ~np.isin(labels, top_classes)
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, model_type in zip(axes, ["unsupervised", "semi_supervised"]):
        embeddings = torch.load(
            CHECKPOINT_DIR / f"{model_type}_embeddings.pt", weights_only=True
        ).numpy()

        print(f"  Computing UMAP projection ({model_type})...")
        umap_2d = _fit_umap(embeddings)

        ax.scatter(umap_2d[other_mask, 0], umap_2d[other_mask, 1],
                   c="lightgray", s=0.5, alpha=0.3, label="Other")

        for i, cls in enumerate(top_classes):
            mask = labels == cls
            name = le.inverse_transform([cls])[0]
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=[cmap(i)], s=1, alpha=0.5, label=name)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        title = model_type.replace("_", "-")
        ax.set_title(title)

    axes[1].legend(markerscale=5, fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_umap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_umap_comparison.png")


