"""Figures for zero-shot prediction results."""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from lilyvae.config import PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR, PAPER_DIR


def plot_r2_vs_observations(output_dir: Path | None = None, model_type: str = "unsupervised"):
    """R² vs number of observations per target variable (fig_r2_vs_observations.png)."""
    output_dir = output_dir or PAPER_DIR

    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    # Load results
    results_path = RESULTS_DIR / "metrics.json"
    if not results_path.exists():
        print("  WARNING: metrics.json not found, skipping R² vs observations plot")
        return

    with open(results_path) as f:
        all_metrics = json.load(f)

    zs_results = all_metrics.get(model_type, {}).get("zeroshot", {})
    if not zs_results:
        print("  WARNING: No zero-shot results found, skipping plot")
        return

    names = []
    n_obs = []
    r2_vals = []

    for name, result in zs_results.items():
        if name in zeroshot_targets:
            names.append(name)
            n_obs.append(zeroshot_targets[name]["n_valid"])
            r2_vals.append(result["r2"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(n_obs, r2_vals, c="steelblue", s=40, edgecolors="black", linewidth=0.5)

    for i, name in enumerate(names):
        ax.annotate(name, (n_obs[i], r2_vals[i]), fontsize=5, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Number of observations")
    ax.set_ylabel("R²")
    ax.set_title("Zero-shot prediction R² vs number of observations")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_r2_vs_observations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_r2_vs_observations.png")


def plot_zeroshot_scatter(output_dir: Path | None = None, model_type: str = "unsupervised"):
    """Scatter plots for best and worst zero-shot predicted variables (fig_zeroshot_scatter.png)."""
    output_dir = output_dir or PAPER_DIR

    results_path = RESULTS_DIR / "metrics.json"
    if not results_path.exists():
        print("  WARNING: metrics.json not found, skipping zero-shot scatter")
        return

    with open(results_path) as f:
        all_metrics = json.load(f)

    zs_results = all_metrics.get(model_type, {}).get("zeroshot", {})
    if not zs_results:
        print("  WARNING: No zero-shot results, skipping plot")
        return

    # Sort by R²
    sorted_vars = sorted(zs_results.items(), key=lambda x: x[1]["r2"], reverse=True)
    # Pick top 3 and bottom 3
    selected = sorted_vars[:3] + sorted_vars[-3:]

    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    prefix = model_type.replace("-", "_")
    embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True)

    from catboost import CatBoostRegressor

    n_plots = len(selected)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, result) in enumerate(selected):
        if i >= len(axes):
            break
        ax = axes[i]
        if name not in zeroshot_targets:
            continue

        target = zeroshot_targets[name]
        mask = target["valid_mask"].numpy().astype(bool)
        z = embeddings.numpy()[mask]
        y = target["values"].numpy()[mask]

        try:
            model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0, random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
            model.fit(z, y)
            y_pred = model.predict(z)
        except Exception:
            ax.text(0.5, 0.5, "CatBoost error", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name} (error)", fontsize=9)
            continue

        n = min(3000, len(y))
        idx = np.random.RandomState(42).choice(len(y), n, replace=False)
        ax.scatter(y[idx], y_pred[idx], alpha=0.2, s=1, c="steelblue")
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=0.5)
        ax.set_title(f"{name} (R²={result['r2']:.3f})", fontsize=9)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_zeroshot_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_zeroshot_scatter.png")
