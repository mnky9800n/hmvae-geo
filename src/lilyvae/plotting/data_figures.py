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
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
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
    """3-panel dataset summary (fig_lily_dataset.png)."""
    output_dir = output_dir or PAPER_DIR
    metadata = load_metadata()

    depth_bins = metadata["depth_bins"]
    borehole_ids = metadata["borehole_ids"]
    depths_m = depth_bins * (BIN_SIZE_CM / 100.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Depth distribution
    axes[0].hist(depths_m, bins=50, color="steelblue", edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Depth (m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Sample depth distribution")

    # (b) Samples per borehole
    bh_counts = pd.Series(borehole_ids).value_counts().values
    axes[1].hist(bh_counts, bins=30, color="coral", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Samples per borehole")
    axes[1].set_ylabel("Number of boreholes")
    axes[1].set_title(f"(b) Samples per borehole (n={len(bh_counts)})")

    # (c) Samples per expedition
    # Extract expedition from borehole_id (first part before "-")
    expeditions = pd.Series(borehole_ids).str.split("-").str[0]
    exp_counts = expeditions.value_counts().sort_index()
    axes[2].bar(range(len(exp_counts)), exp_counts.values, color="seagreen", edgecolor="black", linewidth=0.3)
    axes[2].set_xlabel("Expedition")
    axes[2].set_ylabel("Sample count")
    axes[2].set_title(f"(c) Samples per expedition (n={len(exp_counts)})")
    axes[2].set_xticks(range(len(exp_counts)))
    axes[2].set_xticklabels(exp_counts.index, rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_dataset.png")


def plot_lithology_counts(output_dir: Path | None = None):
    """Bar chart of lithology class counts (fig_lily_lithology_counts.png)."""
    output_dir = output_dir or PAPER_DIR
    labels, le = load_labels()
    label_names = le.inverse_transform(labels.numpy())

    counts = pd.Series(label_names).value_counts()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(counts)), counts.values, color="steelblue", edgecolor="none")
    ax.set_xlabel("Lithology class")
    ax.set_ylabel("Count")
    ax.set_title(f"Lithology class distribution ({len(counts)} classes)")
    ax.set_xticks(range(0, len(counts), max(1, len(counts) // 20)))
    ax.set_xticklabels(
        [counts.index[i] for i in range(0, len(counts), max(1, len(counts) // 20))],
        rotation=90, fontsize=5,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_lithology_counts.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_lithology_counts.png")


def plot_variables_distribution(output_dir: Path | None = None):
    """Histograms of 6 training variables (fig_lily_variables_dist.png).

    Uses raw (unscaled) data loaded from the scaler inverse transform.
    """
    import pickle

    output_dir = output_dir or PAPER_DIR
    features = load_features()

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    raw = scaler.inverse_transform(features.numpy())

    metadata = load_metadata()
    var_names = metadata.get("feature_columns", TRAINING_VARIABLES)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, var_names)):
        ax.hist(raw[:, i], bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_lily_variables_dist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_variables_dist.png")
