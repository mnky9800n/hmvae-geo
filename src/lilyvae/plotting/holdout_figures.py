"""Figure: transfer does not decay as the held-out rock gets less familiar.

The paper asks the reader to accept numbers measured on boreholes the
model never saw, which invites the objection that those boreholes are
not drawn from the training population and the scores are therefore
pessimistic. Correlation coefficients answer that objection but do not
make it visible.

Two panels carry the argument.

The upper panel establishes that the held-out boreholes really are out
of distribution, by comparing each fold's energy distance against the
same statistic computed on random sample-level splits of the same data.
The random split is the no-shift reference: whatever distance it shows
is what this many rows produce by chance alone.

The lower panel shows per-target transfer against that distance, one
small multiple per property, each with an ordinary least squares fit.
Flat fits across the observed range are the result being claimed.
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import torch
from scipy.stats import linregress

from lilyvae.config import PAPER_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from lilyvae.evaluation.folds import Fold, energy_distance
from lilyvae.evaluation.targets import TARGET_REGISTRY
from lilyvae.plotting.style import apply_style

#: Categorical slots 1-4 from the validated reference palette, plus two
#: neutrals. Only two hues actually encode anything here: one for the
#: measured folds and one for the no-shift reference.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
SERIES = "#2a78d6"      # slot 1, blue: measured borehole folds
REFERENCE = "#eb6834"   # slot 2, orange: random-split reference
FIT = "#0b0b0b"

#: Targets worth plotting, in the order they are shown. Chosen as those
#: that transfer at all; a panel of noise around zero would say nothing
#: about whether transfer decays with novelty.
MIN_MEDIAN_R2 = 0.25


def _null_energy_distances(
    features: np.ndarray, n: int = 30, test_fraction: float = 0.1
) -> np.ndarray:
    """Energy distance for random sample-level splits of the same data.

    The reference the borehole folds are measured against. A random split
    breaks no geological structure, so its distance is what this sample
    size yields when nothing is actually shifted.
    """
    rng = np.random.RandomState(SEED)
    n_rows = len(features)
    out = []
    for i in range(n):
        test_mask = np.zeros(n_rows, dtype=bool)
        test_mask[rng.choice(n_rows, int(n_rows * test_fraction), replace=False)] = True
        fold = Fold(
            iteration=i, train_mask=~test_mask, test_mask=test_mask,
            train_boreholes=np.array([]), test_boreholes=np.array([]),
        )
        out.append(energy_distance(fold, features))
    return np.array(out, dtype=float)


def plot_holdout_novelty(
    results_path: Path | None = None,
    output_dir: Path | None = None,
    model: str = "unsupervised_vae",
) -> Path:
    """Render the figure and return where it was written."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_style()
    results_path = results_path or (RESULTS_DIR / "zeroshot_bootstrap.json")
    output_dir = Path(output_dir or PAPER_DIR)
    data = json.loads(Path(results_path).read_text())

    # Per fold: how far out of distribution, and how well each target did.
    distances, per_target = [], {}
    for it in data["per_iteration"]:
        entry = next((m for m in it["models"] if m["model"] == model), None)
        if entry is None:
            continue
        distances.append(it["energy_distance"])
        for name, res in entry.get("per_variable", {}).items():
            per_target.setdefault(name, []).append(res.get("r2", float("nan")))
    distances = np.array(distances, dtype=float)

    summary = data["summary"][model]["per_target"]
    keep = [
        n for n in per_target
        if summary.get(n, {}).get("median", float("-inf")) >= MIN_MEDIAN_R2
    ]
    keep.sort(key=lambda n: summary[n]["median"], reverse=True)

    features = torch.load(
        PROCESSED_DIR / "train_features.pt", weights_only=True
    ).numpy().astype(np.float64)
    null = _null_energy_distances(features, n=len(distances))

    ncols = 3
    nrows = int(np.ceil(len(keep) / ncols))
    fig = plt.figure(figsize=(10.5, 3.0 + 2.5 * nrows))
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[1.15] + [1.0] * nrows, hspace=0.55, wspace=0.28)

    # ---- upper panel: are the held-out boreholes actually unfamiliar? ----
    ax = fig.add_subplot(gs[0, :])
    rng = np.random.RandomState(0)
    ax.scatter(null, np.full_like(null, 1.0) + rng.uniform(-0.11, 0.11, null.size),
               s=34, color=REFERENCE, alpha=0.75, linewidths=0, zorder=3)
    ax.scatter(distances, np.zeros_like(distances) + rng.uniform(-0.11, 0.11, distances.size),
               s=34, color=SERIES, alpha=0.75, linewidths=0, zorder=3)
    for y, vals, colour in ((1.0, null, REFERENCE), (0.0, distances, SERIES)):
        ax.plot([np.median(vals)] * 2, [y - 0.22, y + 0.22],
                color=colour, lw=2.0, solid_capstyle="round", zorder=4)

    ratio = np.median(distances) / max(np.median(null), 1e-12)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["held-out\nboreholes", "random split\n(no shift)"],
                       fontsize=9, color=INK)
    ax.set_xlabel("energy distance from the training rows, in the six input features",
                  fontsize=9, color=MUTED)
    ax.set_ylim(-0.55, 1.55)
    ax.set_title(
        f"Held-out boreholes sit {ratio:.0f}x further from the training data "
        "than a random split does",
        fontsize=11, color=INK, loc="left", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)

    # ---- lower panels: does transfer decay with that distance? ----
    for i, name in enumerate(keep):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        r2 = np.array(per_target[name], dtype=float)
        ok = np.isfinite(r2) & np.isfinite(distances)
        x, y = distances[ok], r2[ok]

        ax.axhline(0.0, color=GRID, lw=0.8, zorder=1)
        ax.scatter(x, y, s=30, color=SERIES, alpha=0.8, linewidths=0, zorder=3)

        note = ""
        if len(x) > 2 and np.ptp(x) > 0:
            fit = linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 50)
            # A fit that cannot be distinguished from flat is drawn dashed,
            # so the eye is not invited to read a trend the data does not
            # support. Only the solid lines are claims.
            significant = fit.pvalue < 0.05
            ax.plot(xs, fit.slope * xs + fit.intercept, color=FIT, lw=2.0,
                    ls="-" if significant else (0, (5, 3)),
                    zorder=4, solid_capstyle="round")
            drop = fit.slope * (x.max() - x.min())
            note = (f"$p$ = {fit.pvalue:.3f}" if significant
                    else f"flat ($p$ = {fit.pvalue:.2f})")
            if significant:
                note += f", falls {abs(drop):.2f}"

        spec = TARGET_REGISTRY.get(name)
        label = spec.label if spec else name
        if spec and spec.tier == "same_modality":
            label += "\u2020"
        ax.set_title(label.replace("Magnetic susceptibility (point)", "Point magnetic susceptibility"), fontsize=9.5, color=INK, loc="left", pad=6)
        ax.text(0.97, 0.06, note, transform=ax.transAxes, ha="right",
                fontsize=8, color=MUTED)
        ax.text(0.03, 0.08, f"median $R^2$ {np.median(y):.2f}",
                transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")

        ax.set_ylim(-0.25, 1.02)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel("transfer $R^2$", fontsize=9, color=MUTED)
        if i // ncols == nrows - 1:
            ax.set_xlabel("energy distance", fontsize=9, color=MUTED)

    n_sig = 0
    for name in keep:
        r2 = np.array(per_target[name], dtype=float)
        ok = np.isfinite(r2) & np.isfinite(distances)
        if len(distances[ok]) > 2 and np.ptp(distances[ok]) > 0:
            n_sig += linregress(distances[ok], r2[ok]).pvalue < 0.05

    fig.suptitle(
        "Transfer changes little across the full range of distribution shift",
        fontsize=13, color=INK, x=0.055, ha="left", y=1.005)
    fig.text(
        0.055, 0.966,
        f"{n_sig} of {len(keep)} properties show a decline distinguishable from "
        "flat; dashed fits are not.   \u2020 same quantity, different instrument.",
        fontsize=9.5, color=MUTED, ha="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_holdout_novelty.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_holdout_novelty())
