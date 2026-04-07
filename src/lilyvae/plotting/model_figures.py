"""Figures for model reconstruction and classification results."""

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from lilyvae.config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
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

    from sklearn.svm import LinearSVC

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = labels.numpy()
    classes = np.unique(y)
    class_counts = np.bincount(y, minlength=max(classes) + 1)
    top_classes = np.argsort(class_counts)[::-1][:10]

    for col_idx, model_type in enumerate(["unsupervised", "semi-supervised"]):
        prefix = model_type.replace("-", "_")
        try:
            embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True)
        except FileNotFoundError:
            continue

        z = embeddings.numpy()
        ax = axes[col_idx]
        aucs = []

        for cls_id in top_classes:
            y_binary = (y == cls_id).astype(int)
            if y_binary.sum() < 2:
                continue
            svm = LinearSVC(max_iter=10000, random_state=42, dual="auto")
            svm.fit(z, y_binary)
            decision = svm.decision_function(z)
            fpr, tpr, _ = roc_curve(y_binary, decision)
            auc_val = float(roc_auc_score(y_binary, decision))
            aucs.append(auc_val)
            class_name = le.inverse_transform([cls_id])[0]
            ax.plot(fpr, tpr, label=f"{class_name} ({auc_val:.2f})", linewidth=1)

        macro_auc = np.mean(aucs) if aucs else float("nan")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_type.title()} (AUC={macro_auc:.3f})")
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


def plot_alpha_sweep(output_dir: Path | None = None):
    """R² and AUC vs supervision weight alpha (fig_alpha_sweep.png)."""
    import json

    output_dir = output_dir or PAPER_DIR

    ablations_path = RESULTS_DIR / "ablations.json"
    if ablations_path.exists():
        with open(ablations_path) as f:
            data = json.load(f)
        if "alpha_weight" in data.get("ablations", {}):
            entries = data["ablations"]["alpha_weight"]
            alphas = [e["value"] for e in entries]
            r2s = [e["r2_avg"] for e in entries]
            aucs = [e["auc_macro"] for e in entries]
        else:
            alphas, r2s, aucs = None, None, None
    else:
        alphas, r2s, aucs = None, None, None

    # Fallback to hardcoded values from the paper table
    if alphas is None:
        alphas = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 500.0, 1000.0]
        r2s = [0.992, 0.992, 0.990, 0.968, 0.984, 0.898, 0.862, 0.831, 0.846, 0.759, 0.770, 0.667, 0.664]
        aucs = [0.950, 0.948, 0.952, 0.953, 0.953, 0.949, 0.949, 0.947, 0.943, 0.949, 0.944, 0.945, 0.942]

    # Use small positive value for log scale at alpha=0
    plot_alphas = [a if a > 0 else 1e-3 for a in alphas]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(plot_alphas, r2s, "o-", color="forestgreen", linewidth=2, markersize=5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Supervision weight $\\alpha$")
    ax1.set_ylabel("$R^2$ (avg)", color="forestgreen")
    ax1.tick_params(axis="y", labelcolor="forestgreen")

    ax2 = ax1.twinx()
    ax2.plot(plot_alphas, aucs, "s-", color="indigo", linewidth=2, markersize=5)
    ax2.set_ylabel("AUC (macro)", color="indigo")
    ax2.set_ylim(0.8, 1.0)
    ax2.tick_params(axis="y", labelcolor="indigo")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_alpha_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_alpha_sweep.png")
