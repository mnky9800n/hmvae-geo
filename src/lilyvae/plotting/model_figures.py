"""Figures for model reconstruction and classification results."""

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from lilyvae.config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    PAPER_DIR,
    TRAINING_VARIABLES,
    INPUT_DIM,
    LATENT_DIM,
    HIDDEN_DIMS,
    BATCH_SIZE,
)
from lilyvae.model.vae import LilyVAE
from lilyvae.data.dataset import get_dataloader
from lilyvae.training.trainer import get_device
from lilyvae.evaluation.classification import evaluate_classification


def _load_vae(model_type: str, device):
    prefix = model_type.replace("-", "_")
    vae = LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)).to(device)
    vae.load_state_dict(torch.load(CHECKPOINT_DIR / f"{prefix}_vae.pt", map_location=device, weights_only=True))
    vae.eval()
    return vae


def _get_reconstructions(vae, dataloader, device):
    all_true, all_pred = [], []
    with torch.no_grad():
        for features, _, _ in dataloader:
            features = features.to(device)
            out = vae(features)
            all_true.append(features.cpu().numpy())
            all_pred.append(out["x_mu"].cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_reconstruction_scatter(output_dir: Path | None = None, device: str = "auto"):
    """Predicted vs true scatter plots for both models (fig_reconstruction_scatter.png)."""
    output_dir = output_dir or PAPER_DIR
    dev = get_device(device)
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    var_names = metadata.get("feature_columns", TRAINING_VARIABLES)

    fig, axes = plt.subplots(len(var_names), 2, figsize=(10, 3 * len(var_names)))

    for col_idx, model_type in enumerate(["unsupervised", "semi-supervised"]):
        try:
            vae = _load_vae(model_type, dev)
        except FileNotFoundError:
            continue

        true, pred = _get_reconstructions(vae, dataloader, dev)

        for row_idx, name in enumerate(var_names):
            ax = axes[row_idx, col_idx]
            # Subsample for plotting
            n = min(5000, len(true))
            idx = np.random.RandomState(42).choice(len(true), n, replace=False)
            ax.scatter(true[idx, row_idx], pred[idx, row_idx], alpha=0.1, s=1, c="steelblue")

            from sklearn.metrics import r2_score
            r2 = r2_score(true[:, row_idx], pred[:, row_idx])
            ax.set_title(f"{name} (R²={r2:.3f})")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")

            # Identity line
            lims = [min(true[:, row_idx].min(), pred[:, row_idx].min()),
                    max(true[:, row_idx].max(), pred[:, row_idx].max())]
            ax.plot(lims, lims, "r--", linewidth=0.5, alpha=0.5)

    axes[0, 0].set_title("Unsupervised\n" + axes[0, 0].get_title())
    axes[0, 1].set_title("Semi-supervised\n" + axes[0, 1].get_title())

    plt.tight_layout()
    plt.savefig(output_dir / "fig_reconstruction_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_reconstruction_scatter.png")


def plot_roc_comparison(output_dir: Path | None = None):
    """ROC curves for top lithologies, both models (fig_roc_comparison.png)."""
    output_dir = output_dir or PAPER_DIR

    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col_idx, model_type in enumerate(["unsupervised", "semi-supervised"]):
        prefix = model_type.replace("-", "_")
        try:
            embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True)
        except FileNotFoundError:
            continue

        cls = evaluate_classification(embeddings, labels, top_n=10)
        ax = axes[col_idx]

        # Plot ROC for top classes
        y_bin = cls["y_bin"]
        decision = cls["decision"]
        classes = cls["classes"]
        class_counts = np.bincount(labels.numpy(), minlength=len(classes))
        top_classes = np.argsort(class_counts)[::-1][:10]

        for idx in top_classes:
            fpr, tpr, _ = roc_curve(y_bin[:, idx], decision[:, idx])
            class_name = le.inverse_transform([classes[idx]])[0]
            auc_val = cls["per_class_auc"].get(int(classes[idx]), float("nan"))
            ax.plot(fpr, tpr, label=f"{class_name} ({auc_val:.2f})", linewidth=1)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_type.title()} (AUC={cls['auc_macro']:.3f})")
        ax.legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_roc_comparison.png")


def plot_generated_variables_grid(output_dir: Path | None = None, device: str = "auto"):
    """PDFs for top lithologies across training variables (fig_generated_variables_grid.png)."""
    import pickle

    output_dir = output_dir or PAPER_DIR
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    var_names = metadata.get("feature_columns", TRAINING_VARIABLES)

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    raw = scaler.inverse_transform(features.numpy())
    y = labels.numpy()

    # Top 5 lithologies
    counts = np.bincount(y)
    top5 = np.argsort(counts)[::-1][:5]
    top5_names = le.inverse_transform(top5)

    fig, axes = plt.subplots(5, len(var_names), figsize=(3 * len(var_names), 12))

    for row, (cls_idx, cls_name) in enumerate(zip(top5, top5_names)):
        mask = y == cls_idx
        for col, name in enumerate(var_names):
            ax = axes[row, col]
            vals = raw[mask, col]
            ax.hist(vals, bins=40, density=True, color="steelblue", alpha=0.7, edgecolor="none")
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_generated_variables_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_generated_variables_grid.png")
