"""Figures describing the LILY dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from lilyvae.config import PROCESSED_DIR, PAPER_DIR, TRAINING_VARIABLES, BIN_SIZE_CM


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


def plot_expedition_map(output_dir: Path | None = None):
    """World map with borehole locations (fig_lily_expedition_map.png)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output_dir = output_dir or PAPER_DIR
    metadata = load_metadata()

    # We need lat/lon from the metadata. If not available, skip.
    if "latitude" not in metadata or "longitude" not in metadata:
        print("  WARNING: lat/lon not in metadata, skipping expedition map")
        return

    lats = metadata["latitude"]
    lons = metadata["longitude"]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=104))
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), s=10, c="red", alpha=0.7, zorder=5)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_expedition_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_expedition_map.png")


def plot_dataset_summary(output_dir: Path | None = None):
    """Composite figure: map background with histogram insets (left) and ship photo (right)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output_dir = output_dir or PAPER_DIR
    metadata = load_metadata()

    depth_bins = metadata["depth_bins"]
    borehole_ids = metadata["borehole_ids"]
    depths_m = depth_bins * (BIN_SIZE_CM / 100.0)

    # Load ship image
    ship_img = plt.imread(Path(__file__).resolve().parents[3] / "Joides.jpg")

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
    inset_w, inset_h = 0.23, 0.14
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
    ax_a.hist(depths_m, bins=50, color="steelblue", edgecolor="black", linewidth=0.3)
    ax_a.set_xlabel("Depth (m)", fontsize=9, labelpad=2)
    ax_a.set_ylabel("Count", fontsize=9, labelpad=2)
    ax_a.set_title("(a) Depth distribution", fontsize=10, pad=3)
    ax_a.tick_params(labelsize=7, pad=1)
    ax_a.patch.set_alpha(0)

    # (b) Samples per borehole
    ax_b = fig.add_axes([inset_x, y_b, inset_w, inset_h], **hist_kw)
    bh_counts = pd.Series(borehole_ids).value_counts().values
    ax_b.hist(bh_counts, bins=30, color="coral", edgecolor="black", linewidth=0.3)
    ax_b.set_xlabel("Samples per borehole", fontsize=9, labelpad=2)
    ax_b.set_ylabel("Boreholes", fontsize=9, labelpad=2)
    ax_b.set_title(f"(b) Samples/borehole (n={len(bh_counts)})", fontsize=10, pad=3)
    ax_b.tick_params(labelsize=7, pad=1)
    ax_b.patch.set_alpha(0)

    # (c) Samples per expedition
    ax_c = fig.add_axes([inset_x, y_c, inset_w, inset_h], **hist_kw)
    expeditions = pd.Series(borehole_ids).str.split("-").str[0]
    exp_counts = expeditions.value_counts().sort_index()
    ax_c.bar(range(len(exp_counts)), exp_counts.values, color="seagreen", edgecolor="black", linewidth=0.3)
    ax_c.set_xlabel("Expedition", fontsize=9, labelpad=2)
    ax_c.set_ylabel("Samples", fontsize=9, labelpad=2)
    ax_c.set_title(f"(c) Samples/expedition (n={len(exp_counts)})", fontsize=10, pad=3)
    ax_c.set_xticks(range(0, len(exp_counts), 2))
    ax_c.set_xticklabels(exp_counts.index[::2], rotation=90, fontsize=4)
    ax_c.tick_params(labelsize=7, pad=1)
    ax_c.patch.set_alpha(0)

    # --- Right upper inset: ship photo with black border ---
    ship_h_over_w = ship_img.shape[0] / ship_img.shape[1]
    ship_w = 0.33
    ship_h = ship_w * ship_h_over_w * (16 / 9)
    ship_x = min(0.74, 0.99 - ship_w)  # clamp so it stays in bounds
    ship_y = 0.90 - ship_h

    ax_ship = fig.add_axes([ship_x, ship_y, ship_w, ship_h], zorder=3)
    ax_ship.imshow(ship_img)
    ax_ship.set_xticks([])
    ax_ship.set_yticks([])
    # Black border via rectangle patch in figure coords
    from matplotlib.patches import Rectangle
    border = Rectangle(
        (ship_x, ship_y), ship_w, ship_h,
        transform=fig.transFigure, linewidth=1.5,
        edgecolor="black", facecolor="none", zorder=4,
    )
    fig.patches.append(border)

    # Caption below the photo -- wrap to exact ship image width
    caption_text = "The drilling ship JOIDES Resolution on Expedition 321"
    wrap_width_px = ship_w * fig.get_size_inches()[0] * fig.dpi
    caption_obj = fig.text(
        ship_x + ship_w / 2, ship_y - 0.01,
        caption_text,
        ha="center", va="top", fontsize=14, fontstyle="italic",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1, alpha=0.92),
    )
    caption_obj._get_wrap_line_width = lambda: wrap_width_px

    plt.savefig(output_dir / "fig_lily_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_dataset.png")


def plot_lithology_counts(output_dir: Path | None = None):
    """Histogram of lithology class sizes on log scale (fig_lily_lithology_counts.png)."""
    output_dir = output_dir or PAPER_DIR
    labels, le = load_labels()
    label_names = le.inverse_transform(labels.numpy())
    counts = pd.Series(label_names).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    log_bins = np.logspace(0, np.log10(counts.max()), 30)
    ax.hist(counts.values, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Number of classes")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_lithology_counts.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_lithology_counts.png")


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
        ax.hist(raw[:, i], bins=80, color=colors.get(name, "steelblue"), edgecolor="none", alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        if name in xlims:
            ax.set_xlim(xlims[name])

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_variables_dist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_variables_dist.png")
