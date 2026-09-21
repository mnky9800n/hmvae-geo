"""Generate the LilyVAE architecture diagram.

Matches the code in src/lilyvae/model/vae.py and the paper description:
 - Input: 6 variables (bulk_density, magnetic_susceptibility, NGR, R, G, B)
 - Learnable mask tokens at 10%% masking (replaces zeros)
 - Encoder: 6 -> 128 -> 64 -> (mu:10, logvar:10)
 - Reparameterization: z = mu + sigma * eps
 - Heteroscedastic decoder: 10 -> 64 -> 128 -> (x_mu:6, x_logvar:6)
 - Optional semi-supervised head: z (10) -> Linear -> 139 softmax
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from lilyvae.config import PAPER_DIR


VAR_COLORS = {
    "Bulk density":      "#f1c40f",
    "Mag. susc.":        "#2c3e50",
    "NGR":               "#8e44ad",
    "R":                 "#e74c3c",
    "G":                 "#27ae60",
    "B":                 "#2980b9",
}
VAR_ORDER = ["Bulk density", "Mag. susc.", "NGR", "R", "G", "B"]
MASKED = {"Mag. susc.", "R"}  # illustrative masked rows

STROKE = "#222222"


def var_box(ax, x, y, w, h, label, color, text_color="white", italic=False):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, edgecolor=STROKE, facecolor=color,
    )
    ax.add_patch(box)
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", color=text_color,
            fontsize=8.5, fontweight="bold", fontstyle=style)


def masked_box(ax, x, y, w, h):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, edgecolor=STROKE, facecolor="#bdc3c7", hatch="///",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, "mask\ntoken",
            ha="center", va="center", color="#111",
            fontsize=7, fontweight="bold")


def trapezoid(ax, x, y, w_left, w_right, h, label, color):
    pts = [
        (x, y + (w_right - w_left) / 2),
        (x, y + h - (w_right - w_left) / 2),
        (x + w, y + h),
        (x + w, y),
    ]
    # Build narrowing or widening trapezoid via left/right heights
    y_center = y + h / 2
    pts = [
        (x, y_center - w_left / 2),
        (x, y_center + w_left / 2),
        (x + w, y_center + w_right / 2),
        (x + w, y_center - w_right / 2),
    ]
    poly = Polygon(pts, closed=True, linewidth=1.0,
                   edgecolor=STROKE, facecolor=color, alpha=0.95)
    ax.add_patch(poly)
    ax.text(x + w / 2, y_center, label,
            ha="center", va="center", fontsize=9, fontweight="bold")


def rect_module(ax, x, y, w, h, label, color, fontsize=9):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, edgecolor=STROKE, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color="#333", style="->,head_length=6,head_width=5"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=1, linewidth=1.0, color=color,
    ))


def band_header(ax, x, y, w, label, fill="#f4f4f4"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, 0.35, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, edgecolor=STROKE, facecolor=fill,
    ))
    ax.text(x + w / 2, y + 0.175, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(-1.2, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Column layout -----
    cols = [
        ("Training Data",   0.3,  1.5),
        ("Standardize",     2.0,  0.9),
        ("Mask (p=0.1)",    3.1,  1.3),
        ("Encoder",         4.6,  2.0),
        ("Latent z",        7.0,  1.6),
        ("Decoder",         9.0,  2.2),
        ("Reconstruction",  11.5, 1.5),
        ("Het. variance",   13.3, 1.4),
    ]
    for name, x, w in cols:
        band_header(ax, x, 8.0, w, name)

    # ----- Training data column -----
    row_h = 0.85
    row_gap = 0.15
    base_y = 7.5 - row_h
    var_xs = {}
    for i, var in enumerate(VAR_ORDER):
        y = base_y - i * (row_h + row_gap)
        var_box(ax, 0.4, y, 1.3, row_h, var, VAR_COLORS[var])
        var_xs[var] = (1.7, y + row_h / 2)

    # Physical / Optical brackets
    ax.text(0.1, 7.5 - row_h / 2 - (row_h + row_gap), "Physical",
            ha="center", va="center", rotation=90, fontsize=8.5, color="#555")
    ax.text(0.1, 7.5 - row_h / 2 - 4 * (row_h + row_gap), "Optical",
            ha="center", va="center", rotation=90, fontsize=8.5, color="#555")

    # ----- Standardize: single 6x1 stack -----
    rect_module(ax, 2.05, 0.5, 0.8, 7.0, "z-score\n(6 dims)", "#ecf0f1", fontsize=9)
    for var in VAR_ORDER:
        _, y = var_xs[var]
        arrow(ax, 1.7, y, 2.05, y)

    # ----- Mask column: learnable mask tokens on 2 of 6 -----
    mask_x = 3.2
    for i, var in enumerate(VAR_ORDER):
        y = 7.5 - row_h - i * (row_h + row_gap)
        if var in MASKED:
            masked_box(ax, mask_x, y, 1.1, row_h)
        else:
            var_box(ax, mask_x, y, 1.1, row_h, var, VAR_COLORS[var])
        arrow(ax, 2.85, y + row_h / 2, mask_x, y + row_h / 2)

    ax.text(3.75, -0.15,
            r"masked entries $\to$ learnable tokens $m_i \in \mathbb{R}^6$",
            ha="center", fontsize=8, color="#333")

    # ----- Encoder trapezoid (narrowing: 6 -> 128 -> 64) -----
    enc_x, enc_w = 4.7, 1.9
    trapezoid(ax, enc_x, 2.0, w_left=5.2, w_right=2.6, h=0, label="", color="#2ecc71")
    # Proper trapezoid:
    enc_center_y = 4.0
    pts = [
        (enc_x, enc_center_y - 2.6),  # bottom-left
        (enc_x, enc_center_y + 2.6),  # top-left
        (enc_x + enc_w, enc_center_y + 1.3),
        (enc_x + enc_w, enc_center_y - 1.3),
    ]
    ax.add_patch(Polygon(pts, closed=True,
                         edgecolor=STROKE, linewidth=1.0,
                         facecolor="#a9dfbf"))
    ax.text(enc_x + enc_w / 2, enc_center_y + 0.3,
            "Encoder", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(enc_x + enc_w / 2, enc_center_y - 0.3,
            "6 → 128 → 64\nReLU", ha="center", va="center",
            fontsize=8)

    # Arrows from each masked row into encoder
    for i, var in enumerate(VAR_ORDER):
        y = 7.5 - row_h / 2 - i * (row_h + row_gap)
        arrow(ax, mask_x + 1.1, y, enc_x, y)

    # ----- mu and logvar blocks -----
    mu_x, mu_y = 6.9, 4.9
    lv_x, lv_y = 6.9, 2.9
    rect_module(ax, mu_x, mu_y, 1.4, 0.9,
                r"$\mu_z$  (10)", "#5dade2", fontsize=9)
    rect_module(ax, lv_x, lv_y, 1.4, 0.9,
                r"$\log\sigma^2_z$  (10)", "#aed6f1", fontsize=9)
    # connect encoder to mu / logvar
    arrow(ax, enc_x + enc_w, enc_center_y + 0.9, mu_x, mu_y + 0.45)
    arrow(ax, enc_x + enc_w, enc_center_y - 0.9, lv_x, lv_y + 0.45)

    # ----- Reparameterization -> z -----
    z_x, z_y = 8.6, 3.9
    rect_module(ax, z_x, z_y, 1.0, 1.1,
                r"$z$" + "\n(10)", "#f5b041", fontsize=10)
    arrow(ax, mu_x + 1.4, mu_y + 0.45, z_x, z_y + 0.9)
    arrow(ax, lv_x + 1.4, lv_y + 0.45, z_x, z_y + 0.2)
    ax.text(z_x + 0.5, z_y - 0.35,
            r"$z = \mu_z + \sigma_z \odot \epsilon$", ha="center",
            fontsize=7.5, color="#333")
    ax.text(z_x + 0.5, z_y - 0.7,
            r"$\epsilon \sim \mathcal{N}(0, I)$", ha="center",
            fontsize=7.5, color="#333")

    # ----- Decoder trapezoid (widening: 10 -> 64 -> 128) -----
    dec_x, dec_w = 10.0, 1.9
    dec_center_y = 4.0
    pts = [
        (dec_x, dec_center_y - 1.3),
        (dec_x, dec_center_y + 1.3),
        (dec_x + dec_w, dec_center_y + 2.6),
        (dec_x + dec_w, dec_center_y - 2.6),
    ]
    ax.add_patch(Polygon(pts, closed=True,
                         edgecolor=STROKE, linewidth=1.0,
                         facecolor="#f5cba7"))
    ax.text(dec_x + dec_w / 2, dec_center_y + 0.3,
            "Decoder", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(dec_x + dec_w / 2, dec_center_y - 0.3,
            "10 → 64 → 128\nReLU", ha="center", va="center",
            fontsize=8)
    arrow(ax, z_x + 1.0, z_y + 0.55, dec_x, dec_center_y)

    # ----- Reconstruction column (mean) -----
    for i, var in enumerate(VAR_ORDER):
        y = 7.5 - row_h - i * (row_h + row_gap)
        var_box(ax, 11.9, y, 1.3, row_h, var + "'", VAR_COLORS[var], italic=True)
        arrow(ax, dec_x + dec_w,
              dec_center_y + (2.5 - i) * 0.6,
              11.9, y + row_h / 2)

    ax.text(12.55, -0.15, r"$\hat{\mu}_x$  (reconstructed mean)",
            ha="center", fontsize=8, color="#333")

    # ----- Heteroscedastic variance column -----
    for i, var in enumerate(VAR_ORDER):
        y = 7.5 - row_h - i * (row_h + row_gap)
        rect_module(ax, 13.4, y, 1.2, row_h,
                    r"$\log\hat{\sigma}^2$", "#fdebd0", fontsize=8.5)
        arrow(ax, dec_x + dec_w,
              dec_center_y + (2.5 - i) * 0.6,
              13.4, y + row_h / 2,
              color="#8c7b5f")

    ax.text(14.0, -0.15, r"clamped to $[-6, 6]$",
            ha="center", fontsize=8, color="#333")

    # ----- Semi-supervised classifier branch -----
    cls_y = 0.2
    ax.plot([z_x + 0.5, z_x + 0.5, 11.5], [z_y, cls_y + 0.4, cls_y + 0.4],
            linestyle=(0, (3, 3)), color="#6c3483", linewidth=1.2)
    rect_module(ax, 11.5, cls_y, 1.4, 0.8,
                "Linear\n10 → 139", "#d7bde2", fontsize=8.5)
    rect_module(ax, 13.1, cls_y, 1.5, 0.8,
                "Softmax\nLithology", "#bb8fce", fontsize=8.5)
    arrow(ax, 12.9, cls_y + 0.4, 13.1, cls_y + 0.4)
    ax.text(10.1, cls_y + 0.4, "semi-supervised only",
            ha="center", va="center", fontsize=7.5,
            color="#6c3483", fontstyle="italic")

    # ----- Loss annotations under the figure -----
    ax.text(8.0, -1.0,
            r"$\mathcal{L} = \mathrm{NLL}(x \mid \hat{\mu}_x, \hat{\sigma}^2_x)"
            r" + \beta \cdot \mathrm{KL}(q(z|x) \,\|\, \mathcal{N}(0, I))"
            r" \;[+\; \alpha \cdot \mathrm{CE}(y, \hat{y})]$",
            ha="center", fontsize=9.5, color="#222")

    fig.tight_layout()
    out = PAPER_DIR / "fig_architecture.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
