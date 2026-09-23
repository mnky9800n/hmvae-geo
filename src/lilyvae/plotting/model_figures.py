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


def plot_alpha_sweep(output_dir: Path | None = None):
    """Effect of the supervision weight, as two panels sharing an x-axis.

    The previous version put reconstruction and classification on a
    twinned y-axis. Two scales on one frame let the reader infer a
    relationship from where the curves cross, which is an artifact of
    the scaling rather than anything in the data, so the panels are
    stacked instead and share only the axis they actually share.

    It also drew from numbers hardcoded in this function, because
    data/results/ablations.json was never produced. It now refuses to
    draw without that file rather than silently plotting constants.
    """
    import json

    import numpy as np

    output_dir = output_dir or PAPER_DIR
    path = RESULTS_DIR / "ablations.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run lilyvae-alpha-sweep` first; this "
            "figure used to fall back to hardcoded values, which is how an "
            "unreproducible figure reached the previous submission."
        )
    entries = json.loads(path.read_text())["ablations"]["alpha_weight"]

    alphas = np.array([e["value"] for e in entries])
    # alpha = 0 cannot be placed on a log axis; it is drawn at a tenth of
    # the smallest non-zero weight and labelled as the unsupervised model.
    positive = alphas[alphas > 0]
    x = np.where(alphas > 0, alphas, positive.min() / 10)

    from lilyvae.plotting.palette import GRID, INK, MUTED, PURPLE, TEAL

    RECON, CLASSIFY = TEAL, PURPLE

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.18})

    panels = [
        (axes[0], "r2_avg", "reconstruction $R^2$", RECON),
        (axes[1], "auc_macro", "lithology AUC", CLASSIFY),
    ]
    for ax, key, label, colour in panels:
        spread = [e.get(f"{key}_spread", {}) for e in entries]
        lo = np.array([s.get("iqr_low", e[key]) for s, e in zip(spread, entries)])
        hi = np.array([s.get("iqr_high", e[key]) for s, e in zip(spread, entries)])
        mid = np.array([e[key] for e in entries])

        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
        ax.plot(x, mid, "o-", color=colour, lw=2.0, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_ylabel(label, fontsize=10, color=MUTED)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    axes[0].set_ylim(0.80, 1.01)
    axes[1].set_ylim(0.80, 1.01)
    axes[0].annotate("unsupervised", xy=(x[0], entries[0]["r2_avg"]),
                     xytext=(x[0], 0.86), ha="center", fontsize=8.5, color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].set_xlabel("$\\alpha$", fontsize=11, color=MUTED)

    # Headings live above the axes rather than on them, so the subtitle
    # cannot land on the top panel's data.
    # Title and reading live in the caption.
    fig.subplots_adjust(top=0.97)

    out = output_dir / "fig_alpha_sweep.png"
    plt.savefig(out, dpi=200, facecolor="white")
    plt.close()
    print(f"  Saved {out.name}")

