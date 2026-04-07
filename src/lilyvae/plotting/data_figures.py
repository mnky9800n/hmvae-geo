"""Figures describing the LILY dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_variable_correlations(output_dir: Path | None = None):
    """Pairwise correlation matrix with histograms on diagonal (fig_lily_correlations.png).

    All training + zero-shot variables. Lower triangle shows Spearman rho as
    colored cells, diagonal shows histograms, upper triangle is blank.
    A box is drawn around the held-out variable block in the bottom-right.
    """
    import pickle
    from matplotlib.patches import Rectangle

    output_dir = output_dir or PAPER_DIR
    features = load_features()
    metadata = load_metadata()

    # Inverse-scale training features to original units
    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    raw_train = scaler.inverse_transform(features.numpy())
    train_names = list(metadata.get("feature_columns", TRAINING_VARIABLES))

    # Load all zero-shot targets
    zeroshot = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    # Build DataFrame: training vars first, then all held-out
    df = pd.DataFrame(raw_train, columns=train_names)

    held_out_names = []
    for key in sorted(zeroshot.keys()):
        if "raw" in key.lower():
            continue
        if "[manual]" in key:
            continue
        if key == "MAD_Bulk density (g/cm^3)":
            continue
        if "Caliper separation" in key:
            continue
        target = zeroshot[key]
        values = target["values"].numpy().copy()
        mask = target["valid_mask"].numpy().astype(bool)
        values[~mask] = np.nan
        # Shorten display name
        short = key.replace("_DataLITH", "")
        df[short] = values
        held_out_names.append(short)

    n_train = len(train_names)
    n_held = len(held_out_names)
    all_names = train_names + held_out_names
    n_vars = len(all_names)

    # Display names: training vars replace _ with space, held-out strip prefix before first _
    def _display_name(name, is_training):
        if is_training:
            return name.replace("_", " ")
        if "_" in name:
            return name.split("_", 1)[1]
        return name
    display_names = [_display_name(n, i < n_train) for i, n in enumerate(all_names)]

    # Compute Spearman correlation matrix
    corr = df[all_names].corr(method="spearman")

    # Create figure with grid of axes
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(28, 28))

    cmap = plt.cm.PRGn
    norm = plt.Normalize(vmin=-1, vmax=1)

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            if i == j:
                # Diagonal: histogram
                vals = df[all_names[i]].dropna().values
                color = "#1f77b4" if i < n_train else "#ff7f0e"
                lo, hi = np.nanpercentile(vals, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    ax.text(0.5, 0.5, "?", ha="center", va="center", transform=ax.transAxes)
                else:
                    vals_clipped = vals[(vals >= lo) & (vals <= hi)]
                    ax.hist(vals_clipped, bins=30, color=color, edgecolor="none", alpha=0.8)
            elif i > j:
                # Lower triangle: colored correlation cell
                r = corr.iloc[i, j]
                if np.isnan(r):
                    ax.set_facecolor("lightgray")
                else:
                    ax.set_facecolor(cmap(norm(r)))
            else:
                # Upper triangle: blank
                ax.axis("off")

        # Row labels on left
        if True:
            axes[i, 0].set_ylabel(display_names[i], fontsize=10, rotation=0, ha="right", va="center", labelpad=3)

    # Column labels on bottom
    for j in range(n_vars):
        axes[-1, j].set_xlabel(display_names[j], fontsize=10, rotation=90, ha="center", va="top", labelpad=3)

    # Draw box around held-out block (bottom-right of lower triangle)
    # In axes coordinates, the held-out block spans rows/cols [n_train:n_vars]
    # Get positions of the corner axes to draw a rectangle in figure coords
    ax_top_left = axes[n_train, n_train]
    ax_bot_right = axes[n_vars - 1, n_vars - 1]
    fig.canvas.draw()

    pos_tl = ax_top_left.get_position()
    pos_br = ax_bot_right.get_position()

    rect = Rectangle(
        (pos_tl.x0, pos_br.y0),
        pos_br.x1 - pos_tl.x0,
        pos_tl.y1 - pos_br.y0,
        transform=fig.transFigure,
        linewidth=3, edgecolor="red", facecolor="none",
        clip_on=False, zorder=10,
    )
    fig.patches.append(rect)

    # Label
    fig.text(
        (pos_tl.x0 + pos_br.x1) / 2, pos_tl.y1 + 0.005,
        "held-out (zero-shot) variables",
        ha="center", va="bottom", fontsize=36, fontweight="bold", color="red",
    )

    # Box around training variables (top-left block)
    ax_train_tl = axes[0, 0]
    ax_train_br = axes[n_train - 1, n_train - 1]
    pos_train_tl = ax_train_tl.get_position()
    pos_train_br = ax_train_br.get_position()
    rect_train = Rectangle(
        (pos_train_tl.x0, pos_train_br.y0),
        pos_train_br.x1 - pos_train_tl.x0,
        pos_train_tl.y1 - pos_train_br.y0,
        transform=fig.transFigure,
        linewidth=3, edgecolor="blue", facecolor="none",
        clip_on=False, zorder=10,
    )
    fig.patches.append(rect_train)
    fig.text(
        pos_train_br.x1 + 0.005, (pos_train_tl.y1 + pos_train_br.y0) / 2,
        "training variables",
        ha="left", va="center", fontsize=36, fontweight="bold", color="blue",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, aspect=30)
    cbar.set_label("Spearman $\\rho$", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_dir / "fig_lily_correlations.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig_lily_correlations.png")
