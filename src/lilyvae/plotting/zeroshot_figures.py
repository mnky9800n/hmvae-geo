"""Figures for zero-shot prediction results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lilyvae.config import RESULTS_DIR, PAPER_DIR
from lilyvae.plotting.style import apply_style


def plot_r2_unsup_vs_semi(output_dir: Path | None = None):
    """Scatter: unsupervised vs semi-supervised zero-shot R² per variable (fig_r2_unsup_vs_semi.png)."""
    apply_style()
    output_dir = output_dir or PAPER_DIR

    results_path = RESULTS_DIR / "zeroshot_fulldata.json"
    if not results_path.exists():
        print("  WARNING: zeroshot_fulldata.json not found, skipping")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    unsup_zs = all_results.get("unsupervised", {}).get("per_variable", {})
    semi_zs = all_results.get("semi_supervised", {}).get("per_variable", {})

    if not unsup_zs or not semi_zs:
        print("  WARNING: missing zero-shot results for one or both models")
        return

    # Collect paired R² values
    names = []
    unsup_r2 = []
    semi_r2 = []
    for name in unsup_zs:
        if name in semi_zs:
            u = unsup_zs[name]["r2"]
            s = semi_zs[name]["r2"]
            if np.isfinite(u) and np.isfinite(s) and u > -1 and s > -1:
                names.append(name)
                unsup_r2.append(u)
                semi_r2.append(s)

    unsup_r2 = np.array(unsup_r2)
    semi_r2 = np.array(semi_r2)

    # Categorize variables by measurement type
    categories = {}
    for name in names:
        nl = name.lower()
        if any(x in nl for x in ["declination", "inclination"]):
            categories[name] = "Orientation"
        elif any(x in nl for x in ["tcon", "thermal"]):
            categories[name] = "Thermal conductivity"
        elif any(x in nl for x in ["pw", "p-wave", "sonic", "caliper"]):
            categories[name] = "Acoustic / P-wave"
        elif any(x in nl for x in ["rsc", "reflectance", "tristimulus", "spectral"]):
            categories[name] = "Reflectance / color"
        elif any(x in nl for x in ["mad", "porosity", "density", "moisture", "void", "mass", "vol"]):
            categories[name] = "Physical properties (MAD)"
        elif any(x in nl for x in ["srm", "magnetic moment", "intensity", "demag"]):
            categories[name] = "Magnetic (SRM)"
        elif "msp" in nl:
            categories[name] = "Point mag. susceptibility"
        else:
            categories[name] = "Other"

    cat_colors = {
        "Thermal conductivity": "#e41a1c",
        "Acoustic / P-wave": "#377eb8",
        "Reflectance / color": "#4daf4a",
        "Physical properties (MAD)": "#ff7f00",
        "Magnetic (SRM)": "#984ea3",
        "Point mag. susceptibility": "#a65628",
        "Orientation": "#999999",
        "Other": "#666666",
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    for cat, color in cat_colors.items():
        mask = [categories[n] == cat for n in names]
        if not any(mask):
            continue
        x = semi_r2[mask]
        y = unsup_r2[mask]
        ax.scatter(x, y, c=color, s=35, edgecolors="black", linewidth=0.3, alpha=0.85, label=cat)

    # Identity line
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Semi-supervised zero-shot $R^2$")
    ax.set_ylabel("Unsupervised zero-shot $R^2$")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_r2_unsup_vs_semi.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig_r2_unsup_vs_semi.png")
