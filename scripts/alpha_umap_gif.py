"""Train VAE at multiple alpha values, compute UMAP, render animated GIF.

Left panel: static unsupervised UMAP.
Right panel: semi-supervised UMAP morphing as alpha increases.

Run on lambda-scalar:
    cd ~/repos/hmvae-geo && .venv/bin/python scripts/alpha_umap_gif.py
"""

import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import umap
import io

from lilyvae.config import (
    INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM, MASK_FRACTION,
    BETA_START, BETA_END, BETA_ANNEAL_EPOCHS, BATCH_SIZE, SEED,
    PROCESSED_DIR, CHECKPOINT_DIR, PAPER_DIR,
)
from lilyvae.data.dataset import get_dataloader
from lilyvae.model.vae import LilyVAE
from lilyvae.model.heads import LithologyClassifier
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.loss import (
    heteroscedastic_reconstruction_loss,
    kl_divergence,
    beta_schedule,
)
from lilyvae.training.trainer import get_device

ALPHA_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 500.0, 1000.0]
NUM_LITHOLOGY_CLASSES = 139
ALPHA_ANNEAL_EPOCHS = 50


def train_and_embed(alpha: float, dataloader, device) -> np.ndarray:
    """Train a VAE at the given alpha and return embeddings."""
    torch.manual_seed(SEED)

    vae = LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)).to(device)
    masking = LearnableMasking(INPUT_DIM, MASK_FRACTION).to(device)
    params = list(vae.parameters()) + list(masking.parameters())

    classifier = None
    if alpha > 0:
        classifier = LithologyClassifier(LATENT_DIM, NUM_LITHOLOGY_CLASSES).to(device)
        params += list(classifier.parameters())

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        vae.train()
        masking.train()
        if classifier:
            classifier.train()

        beta = beta_schedule(epoch, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)

        if classifier:
            cur_alpha = alpha * min(1.0, epoch / ALPHA_ANNEAL_EPOCHS)
        else:
            cur_alpha = 0.0

        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)
            masked = masking(features)
            out = vae(masked)

            recon = heteroscedastic_reconstruction_loss(features, out["x_mu"], out["x_logvar"])
            kl = kl_divergence(out["z_mu"], out["z_logvar"])
            loss = recon + beta * kl

            if classifier:
                logits = classifier(out["z"])
                cls_loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss + cur_alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  alpha={alpha:.2f} epoch {epoch+1}/{NUM_EPOCHS}")

    # Extract embeddings
    vae.eval()
    all_z = []
    with torch.no_grad():
        for features, _, _ in dataloader:
            z_mu = vae.get_embeddings(features.to(device))
            all_z.append(z_mu.cpu())
    return torch.cat(all_z, dim=0).numpy()


def main():
    device = get_device("auto")
    print(f"Device: {device}")

    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    # Load labels and label encoder for coloring
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    counts = np.bincount(labels)
    top_n = 10
    top_classes = np.argsort(counts)[::-1][:top_n]
    other_mask = ~np.isin(labels, top_classes)
    cmap = plt.cm.tab10

    # --- Unsupervised (static, alpha=0) ---
    print("Computing unsupervised embeddings + UMAP...")
    unsup_emb = torch.load(CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True).numpy()
    unsup_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(unsup_emb)

    # --- Train at each alpha and compute UMAP ---
    alpha_umaps = {}
    for alpha in ALPHA_VALUES:
        print(f"\n=== Training alpha={alpha} ===")
        if alpha == 0.0:
            emb = unsup_emb
        else:
            emb = train_and_embed(alpha, dataloader, device)
        print(f"  Computing UMAP for alpha={alpha}...")
        u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(emb)
        # Flip axes to match unsupervised orientation
        if alpha > 0:
            u[:, 0], u[:, 1] = u[:, 1].copy(), u[:, 0].copy()
        alpha_umaps[alpha] = u

    # --- Render frames ---
    print("\nRendering frames...")
    frames = []
    for alpha in ALPHA_VALUES:
        u = alpha_umaps[alpha]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: static unsupervised
        ax = axes[0]
        ax.scatter(unsup_umap[other_mask, 0], unsup_umap[other_mask, 1],
                   c="lightgray", s=0.5, alpha=0.3)
        for i, cls in enumerate(top_classes):
            mask = labels == cls
            name = le.inverse_transform([cls])[0]
            ax.scatter(unsup_umap[mask, 0], unsup_umap[mask, 1],
                       c=[cmap(i)], s=1, alpha=0.5, label=name)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Unsupervised ($\\alpha=0$)")

        # Right: semi-supervised at this alpha
        ax = axes[1]
        ax.scatter(u[other_mask, 0], u[other_mask, 1],
                   c="lightgray", s=0.5, alpha=0.3)
        for i, cls in enumerate(top_classes):
            mask = labels == cls
            name = le.inverse_transform([cls])[0]
            ax.scatter(u[mask, 0], u[mask, 1],
                       c=[cmap(i)], s=1, alpha=0.5, label=name)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        if alpha == 0:
            ax.set_title("Semi-supervised ($\\alpha=0$, same as unsupervised)")
        else:
            ax.set_title(f"Semi-supervised ($\\alpha={alpha}$)")

        axes[1].legend(markerscale=5, fontsize=7, loc="best")
        plt.tight_layout()

        # Render to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    # --- Save GIF ---
    out_path = PAPER_DIR / "fig_umap_alpha_sweep.gif"
    # Hold each frame for 1.5s, hold first and last longer
    durations = [2500] + [1500] * (len(frames) - 2) + [4000]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
