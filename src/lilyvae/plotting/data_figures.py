"""Figures describing the LILY dataset."""

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path

from lilyvae.config import PROCESSED_DIR, PAPER_DIR, TRAINING_VARIABLES, BIN_SIZE_CM
from lilyvae.plotting.palette import TEAL


def load_metadata():
    return torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)


def load_features():
    return torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)


def load_labels():
    import pickle
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    return labels, le


def plot_dataset_summary(output_dir: Path | None = None):
    """Global distribution of the modelled subset (fig_lily_dataset.png).

    The map fills the frame. A previous version gave a third of the
    figure to a photograph of the drill ship and shrank the distribution
    to fit around it, which inverted the priority: the point of the
    figure is where the data is and, as importantly, where it is not.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output_dir = output_dir or PAPER_DIR
    metadata = load_metadata()

    depth_bins = metadata["depth_bins"]
    borehole_ids = metadata["borehole_ids"]
    depths_m = depth_bins * (BIN_SIZE_CM / 100.0)

    # Load ship image

    fig = plt.figure(figsize=(16, 9))

    # --- Full-figure map as background ---
    ax_map = fig.add_axes([0, 0, 1, 1], projection=ccrs.Robinson(central_longitude=104))
    ax_map.set_global()
    from cartopy.io.img_tiles import GoogleTiles
    satellite_tiles = GoogleTiles(style="satellite")
    ax_map.add_image(satellite_tiles, 4)
    lats = metadata.get("latitude")
    lons = metadata.get("longitude")
    if lats is not None and lons is not None:
        ax_map.scatter(lons, lats, transform=ccrs.PlateCarree(), s=10, c="red", alpha=0.7, zorder=5)

    # --- Left insets: 3 histograms in a white box ---
    # Inset axes are placed with generous margins inside the box
    inset_w, inset_h = 0.185, 0.115
    inset_x = -0.06  # left margin inside box (shifted 10% left)
    gap = 0.08  # vertical space between axes (for xlabel + next title)

    # Positions top-down
    y_a = 0.78
    y_b = y_a - inset_h - gap
    y_c = y_b - inset_h - gap

    # White box covers everything including titles above and labels below
    from matplotlib.patches import FancyBboxPatch
    box_x = -0.095
    box_y = y_c - 0.05  # room below bottom axis for xlabel + ticks
    box_w = inset_w + 0.07
    box_h = (y_a + inset_h + 0.04) - box_y  # room above top axis for title
    bg_box = FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0.01",
        facecolor="white", edgecolor="black", linewidth=1.5, alpha=0.92,
        transform=fig.transFigure, zorder=2,
    )
    fig.patches.append(bg_box)

    hist_kw = dict(facecolor="white", zorder=3)

    # (a) Depth distribution
    ax_a = fig.add_axes([inset_x, y_a, inset_w, inset_h], **hist_kw)
    ax_a.hist(depths_m, bins=50, color=TEAL, edgecolor="black", linewidth=0.3)
    ax_a.set_xlabel("Depth (m)", fontsize=9, labelpad=2)
    ax_a.set_ylabel("Count", fontsize=9, labelpad=2)
    ax_a.set_title("(a) Depth distribution", fontsize=10, pad=3)
    ax_a.tick_params(labelsize=7, pad=1)
    ax_a.patch.set_alpha(0)

    # (b) Samples per borehole
    ax_b = fig.add_axes([inset_x, y_b, inset_w, inset_h], **hist_kw)
    bh_counts = pd.Series(borehole_ids).value_counts().values
    ax_b.hist(bh_counts, bins=30, color=TEAL, edgecolor="black", linewidth=0.3)
    ax_b.set_xlabel("Samples per borehole", fontsize=9, labelpad=2)
    ax_b.set_ylabel("Boreholes", fontsize=9, labelpad=2)
    ax_b.set_title(f"(b) Samples/borehole (n={len(bh_counts)})", fontsize=10, pad=3)
    ax_b.tick_params(labelsize=7, pad=1)
    ax_b.patch.set_alpha(0)

    # (c) Samples per expedition
    ax_c = fig.add_axes([inset_x, y_c, inset_w, inset_h], **hist_kw)
    expeditions = pd.Series(borehole_ids).str.split("-").str[0]
    exp_counts = expeditions.value_counts().sort_index()
    ax_c.bar(range(len(exp_counts)), exp_counts.values, color=TEAL, edgecolor="black", linewidth=0.3)
    ax_c.set_xlabel("Expedition", fontsize=9, labelpad=2)
    ax_c.set_ylabel("Samples", fontsize=9, labelpad=2)
    ax_c.set_title(f"(c) Samples/expedition (n={len(exp_counts)})", fontsize=10, pad=3)
    ax_c.set_xticks(range(0, len(exp_counts), 2))
    ax_c.set_xticklabels(exp_counts.index[::2], rotation=90, fontsize=4)
    ax_c.tick_params(labelsize=7, pad=1)
    ax_c.patch.set_alpha(0)

    # The ship photograph that used to sit here has been removed. It was
    # decorative, it occupied a third of the frame that the global
    # distribution needed, and it implied a single platform: LILY draws on
    # holes drilled from several vessels, and the same measurements are
    # made on continental cores that never go to sea.

    plt.savefig(output_dir / "fig_lily_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_dataset.png")


def plot_variables_distribution(output_dir: Path | None = None):
    """Histograms of 6 training variables (fig_lily_variables_dist.png)."""
    import pickle

    output_dir = output_dir or PAPER_DIR
    features = load_features()

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    raw = scaler.inverse_transform(features.numpy())

    metadata = load_metadata()
    var_names = metadata.get("feature_columns", TRAINING_VARIABLES)

    colors = {
        "bulk_density": "#FF6F00",
        "magnetic_susceptibility": "#AB47BC",
        "NGR": "#00BFA5",
        "R": "#E53935",
        "G": "#43A047",
        "B": "#1E88E5",
    }
    xlims = {
        "bulk_density": (0, 4),
        "NGR": (0, 100),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, var_names)):
        ax.hist(raw[:, i], bins=80, color=TEAL, edgecolor="none", alpha=0.85)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        if name in xlims:
            ax.set_xlim(xlims[name])

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_variables_dist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_variables_dist.png")


def plot_variable_correlations(output_dir: Path | None = None):
    """Which inputs carry which held-out variable (fig_lily_correlations.png).

    This replaces a full pairwise grid over every variable, which at
    roughly sixty variables meant several thousand panels on a 28-inch
    canvas and was unreadable without magnification. The content that
    mattered was always one block of it: the six inputs against the
    variables the model is asked to predict.

    Columns are restricted to the tiers the paper reports, so the
    procedural columns and the algebraic restatements of one another are
    not shown. Showing them exaggerates how much is being predicted, and
    a reader counting columns is the person most likely to be misled.

    The scale is banded rather than continuous, because a correlation
    matrix is read by matching a cell to a number.
    """
    import pickle

    from lilyvae.evaluation.targets import (
        REPORTED_TIERS, TARGET_REGISTRY, select_targets,
    )
    from lilyvae.plotting.palette import GRID, INK, MUTED, diverging_cmap

    output_dir = output_dir or PAPER_DIR
    features = load_features()
    metadata = load_metadata()

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    raw = scaler.inverse_transform(features.numpy())
    train_names = list(metadata.get("feature_columns", TRAINING_VARIABLES))

    zeroshot = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    targets = select_targets(zeroshot, REPORTED_TIERS)
    targets.sort(key=lambda n: (TARGET_REGISTRY[n].tier != "independent",
                                TARGET_REGISTRY[n].label))

    corr = np.full((len(train_names), len(targets)), np.nan)
    for j, name in enumerate(targets):
        values = zeroshot[name]["values"].numpy().astype(float).copy()
        values[~zeroshot[name]["valid_mask"].numpy().astype(bool)] = np.nan
        for i in range(len(train_names)):
            both = np.isfinite(values) & np.isfinite(raw[:, i])
            if both.sum() > 50:
                corr[i, j] = spearmanr(raw[both, i], values[both]).statistic

    cmap, norm, bounds = diverging_cmap(20)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(targets), 4.6))
    ax.imshow(corr, cmap=cmap, norm=norm, aspect="auto")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if not np.isfinite(corr[i, j]):
                continue
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.0,
                    color="white" if abs(corr[i, j]) > 0.55 else INK)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(
        [TARGET_REGISTRY[n].label
         + ("\u2020" if TARGET_REGISTRY[n].tier == "same_modality" else "")
         for n in targets],
        rotation=40, ha="right", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(train_names)))
    ax.set_yticklabels([n.replace("_", " ") for n in train_names],
                       fontsize=10, color=INK)
    ax.set_xticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(train_names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                       ticks=bounds[::4], pad=0.02, fraction=0.04)
    bar.set_label("Spearman $\\rho$", fontsize=9.5, color=MUTED)
    bar.ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
    bar.outline.set_visible(False)

    # Two lines of heading above the axes, the second clear of the first.
    ax.set_title(
        "Rank correlation between the six inputs and the variables predicted "
        "from them",
        fontsize=12, color=INK, loc="left", pad=44)
    ax.text(0.0, 1.0,
            "\u2020 the same quantity as an input, measured by another "
            "instrument.\nProcedural columns and algebraic restatements of "
            "other targets are not shown.",
            fontsize=9, color=MUTED, ha="left", va="bottom",
            linespacing=1.5, transform=ax.transAxes)

    out = output_dir / "fig_lily_correlations.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out.name}")

