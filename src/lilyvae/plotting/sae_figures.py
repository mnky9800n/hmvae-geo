"""Figure: the physics the literature names is not a set of axes here.

Section 4.3 writes five couplings down before looking, each with a
published source, and scores every one against a measured held-out
variable. The result is negative, and a negative result stated in prose
asks the reader to take the author's word for what "no better than
chance" means. This figure shows the chance.

Each row is one hypothesis. The grey band is what an arbitrary direction
through the same embedding reaches: the latent is rotated by a random
orthogonal matrix, which preserves every distance and correlation in it
and destroys only the particular axes the sparse fit chose, and the band
runs from zero to the 95th percentile of thirty such rotations. The tick
inside it is their median. The dot is what the named hypothesis scored.

A recovered hypothesis would put its dot past the right edge of its band.
None does. Rows are ordered by how far the dot sits above the rotation
median, so the one positive case (compaction) is at the top and is still
inside its band.

The left panel is the strongest correlation between any of the 32 sparse
features and the target, which is a maximum over features and high by
construction. The right panel is selectivity, the gap between a feature's
best target and its second best, which a shared signal cannot inflate.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np

from lilyvae.config import PAPER_DIR, RESULTS_DIR
from lilyvae.plotting.palette import GRID, INK, MUTED, TEAL
from lilyvae.plotting.style import apply_style

#: The JSON carries the LILY column name; a figure needs the property.
TARGET_NAMES = {
    "MAD_Porosity (vol%)": "porosity",
    "TCON_Thermal conductivity mean (W/(m*K))": "thermal conductivity",
    "PWC_P-wave velocity x (m/s)": "P-wave velocity",
    "SRM_Intensity raw (A/m)": "magnetisation",
    "MAD_Grain density (g/cm^3)": "grain density",
}

HYPOTHESIS_NAMES = {"grain_framework": "grain framework"}


def _panel(ax, rows, observed_key, null_key, p95_key, xlabel, xmax):
    """One bullet panel: rotation band, its median, the observed dot."""
    ys = np.arange(len(rows))[::-1]

    for y, r in zip(ys, rows):
        ax.barh(y, r[p95_key], height=0.52, color=GRID, edgecolor="white",
                linewidth=1.0, zorder=2)
        ax.plot([r[null_key]] * 2, [y - 0.26, y + 0.26], color=MUTED,
                lw=1.4, zorder=3, solid_capstyle="butt")
        ax.plot(r[observed_key], y, "o", ms=9, color=TEAL,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        # Just clear of the band, so the number reads as the dot's and
        # not as the row above it.
        ax.text(r[observed_key], y + 0.30, f"{r[observed_key]:.2f}",
                ha="center", va="bottom", fontsize=8.2, color=INK)

    ax.set_yticks(ys)
    ax.set_ylim(-0.62, len(rows) - 0.42)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return ys


def plot_sae_hypotheses(
    results_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Draw the hypothesis test as measured values against their null."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    output_dir = Path(output_dir or PAPER_DIR)
    results_path = Path(
        results_path or RESULTS_DIR / "sae_hypotheses_selective.json"
    )
    with open(results_path) as f:
        payload = json.load(f)

    rows = sorted(payload["results"],
                  key=lambda r: r["observed_rho"] - r["null_rho"],
                  reverse=True)
    labels = [
        (HYPOTHESIS_NAMES.get(r["hypothesis"], r["hypothesis"]),
         TARGET_NAMES.get(r["target"], r["target"]))
        for r in rows
    ]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9.6, 0.78 * len(rows) + 2.2), sharey=True,
        gridspec_kw={"wspace": 0.12},
    )

    ys = _panel(ax_l, rows, "observed_rho", "null_rho", "null_p95",
                "strongest $|\\rho|$ with the target", 0.92)
    _panel(ax_r, rows, "selectivity", "selectivity_null",
           "selectivity_null_p95",
           "selectivity (best target minus second best)", 0.58)

    ax_l.set_yticklabels([f"{h}\n{t}" for h, t in labels], fontsize=9.6,
                         color=INK, linespacing=1.35)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=8, color=TEAL,
               markeredgecolor="white", label="the named hypothesis"),
        Patch(facecolor=GRID, edgecolor="white",
              label="random rotations, to their 95th percentile"),
        Line2D([], [], color=MUTED, lw=1.4, label="random rotations, median"),
    ]
    ax_l.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.18),
                ncols=3, frameon=False, fontsize=9, labelcolor=MUTED,
                handlelength=1.4, columnspacing=1.4)

    # Title and reading live in the caption.
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_sae_hypotheses.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_sae_hypotheses())
