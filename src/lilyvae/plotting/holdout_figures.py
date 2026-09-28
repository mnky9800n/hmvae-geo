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
from lilyvae.plotting.palette import (
    ACCENT, GRID, INK, MUTED, PURPLE, SEQUENTIAL, TEAL,
)

SERIES = TEAL        # the measured thing
REFERENCE = PURPLE   # what it is being compared against
FIT = INK

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

    # Energy distance has no natural scale, so mark what a known
    # displacement of the same data looks like. Calibrated by shifting a
    # single standardised input variable on a random tenth of the rows:
    # the statistic grows roughly as the square of the displacement.
    calibration = [(0.0213, "0.25 SD"), (0.0766, "0.5 SD"), (0.3086, "1 SD")]
    for value, label in calibration:
        if value > distances.max() * 1.15:
            continue
        ax.axvline(value, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=2)
        ax.text(value, 1.60, label, fontsize=7.5, color=MUTED,
                ha="center", va="bottom")
    # Caption sits on its own line above the tick labels so the two do
    # not overlap at the left edge.
    ax.text(0.0, 1.10, "dotted lines: energy distance equivalent to shifting "
            "one input variable by",
            fontsize=7.5, color=MUTED, ha="left", va="bottom",
            transform=ax.transAxes)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["held-out\nboreholes", "random split\n(no shift)"],
                       fontsize=9, color=INK)
    ax.set_xlabel("energy distance", fontsize=10, color=MUTED)
    ax.set_ylim(-0.55, 1.85)
    ax.set_title(f"{ratio:.0f}x further than a random split",
                 fontsize=10, color=MUTED, loc="left", pad=24)
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

    # Title and reading live in the caption.
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_holdout_novelty.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_holdout_novelty())


#: A lithology holding less than this share of the training core counts
#: as one the model barely saw.
RARE_IN_TRAINING = 0.005


def fold_novelty(borehole_ids: np.ndarray, labels: np.ndarray,
                 n_folds: int = 30, test_fraction: float = 0.1) -> np.ndarray:
    """Share of each fold's held-out core that is rock the model barely saw.

    Energy distance is the statistically proper measure of how far a
    held-out set sits from training, and it is unreadable to anyone who
    has not met it. This is the same question asked in units a core
    describer already thinks in: what fraction of the rock being
    predicted is a lithology that made up almost none of the training
    material.
    """
    from lilyvae.evaluation.folds import borehole_splits

    out = []
    for fold in borehole_splits(borehole_ids, n_folds, test_fraction, seed=SEED):
        train_labels, test_labels = labels[fold.train_mask], labels[fold.test_mask]
        share = np.bincount(train_labels, minlength=int(labels.max()) + 1)
        share = share / max(len(train_labels), 1)
        out.append(float(np.mean(share[test_labels] < RARE_IN_TRAINING)))
    return np.array(out)


def plot_transfer_by_novelty(
    results_path: Path | None = None,
    output_dir: Path | None = None,
    model: str = "unsupervised_vae",
) -> Path:
    """The same finding as `plot_holdout_novelty`, for a reader in a hurry.

    Folds are split into the third whose held-out core most resembles the
    training material and the third that least resembles it, and each
    property's accuracy is shown for both. A short connector means the
    property is predicted about as well on unfamiliar rock as on
    familiar rock, which is the whole claim.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    results_path = results_path or (RESULTS_DIR / "zeroshot_bootstrap.json")
    output_dir = Path(output_dir or PAPER_DIR)
    data = json.loads(Path(results_path).read_text())

    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    novelty = fold_novelty(np.asarray(metadata["borehole_ids"]), labels,
                           n_folds=len(data["per_iteration"]))

    per_target = {}
    for i, it in enumerate(data["per_iteration"]):
        entry = next((m for m in it["models"] if m["model"] == model), None)
        if entry is None:
            continue
        for name, res in entry.get("per_variable", {}).items():
            per_target.setdefault(name, []).append((i, res.get("r2", float("nan"))))

    lo, hi = np.percentile(novelty, [33.3, 66.7])
    familiar = novelty <= lo
    unfamiliar = novelty >= hi

    summary = data["summary"][model]["per_target"]
    keep = [n for n in per_target
            if summary.get(n, {}).get("median", float("-inf")) >= MIN_MEDIAN_R2]
    keep.sort(key=lambda n: summary[n]["median"])

    rows = []
    for name in keep:
        vals = np.full(len(novelty), np.nan)
        for i, r2 in per_target[name]:
            vals[i] = r2
        a = np.nanmedian(vals[familiar])
        b = np.nanmedian(vals[unfamiliar])
        spec = TARGET_REGISTRY.get(name)
        label = spec.label if spec else name
        label = label.replace("Magnetic susceptibility (point)",
                              "Point magnetic susceptibility")
        rows.append((label, a, b))

    fig, ax = plt.subplots(figsize=(9.2, 0.46 * len(rows) + 2.6))
    ys = np.arange(len(rows))

    for y, (_, a, b) in zip(ys, rows):
        ax.plot([a, b], [y, y], color=GRID, lw=3.0, solid_capstyle="round", zorder=2)
    ax.scatter([r[1] for r in rows], ys, s=110, color=SERIES,
               zorder=4, linewidths=2, edgecolors="white")
    ax.scatter([r[2] for r in rows], ys, s=110, color=REFERENCE,
               zorder=4, linewidths=2, edgecolors="white")

    # Each value sits on the outer side of its own dot, so a number is
    # always read against the dot it belongs to. Labelling by which value
    # is larger instead makes the colour-to-number mapping flip from row
    # to row, which is exactly how a reader loses track of the encoding.
    for y, (_, a, b) in zip(ys, rows):
        a_left = a <= b
        ax.text(a - 0.035 if a_left else a + 0.035, y, f"{a:.2f}",
                ha="right" if a_left else "left", va="center",
                fontsize=9.5, color=INK)
        ax.text(b + 0.035 if a_left else b - 0.035, y, f"{b:.2f}",
                ha="left" if a_left else "right", va="center",
                fontsize=9.5, color=INK)

    # Name the two dots where the reader first meets them.
    top = len(rows) - 1
    a_top, b_top = rows[top][1], rows[top][2]
    ax.annotate("familiar rock", xy=(a_top, top), xytext=(a_top, top + 0.42),
                ha="center", va="bottom", fontsize=9, color=SERIES,
                arrowprops=dict(arrowstyle="-", color=SERIES, lw=1.0,
                                shrinkA=0, shrinkB=7))
    ax.annotate("unfamiliar rock", xy=(b_top, top), xytext=(b_top - 0.10, top + 0.42),
                ha="center", va="bottom", fontsize=9, color=REFERENCE,
                arrowprops=dict(arrowstyle="-", color=REFERENCE, lw=1.0,
                                shrinkA=0, shrinkB=7))

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.6, len(rows) + 0.15)
    ax.set_xlabel("how much of the property the model gets right on boreholes "
                  "it never saw  ($R^2$)", fontsize=10, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", markersize=10, color=SERIES,
                   label=f"familiar rock: boreholes whose core is "
                         f"{novelty[familiar].mean()*100:.0f}% rock types "
                         "the model barely saw"),
            Line2D([], [], marker="o", ls="", markersize=10, color=REFERENCE,
                   label=f"unfamiliar rock: "
                         f"{novelty[unfamiliar].mean()*100:.0f}% rock types "
                         "the model barely saw"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=1,
        frameon=False, fontsize=9.5, labelcolor=MUTED,
        handletextpad=0.4, columnspacing=2.4,
    )

    fig.suptitle("The model does about as well on unfamiliar rock as on familiar rock",
                 fontsize=13.5, color=INK, x=0.012, ha="left", y=1.005)
    better = sum(1 for _, a, b in rows if b >= a)
    fig.text(0.012, 0.93,
             "Each borehole is predicted by a model that never saw it. Folds are "
             "split by how much of the held-out core\nis a rock type that made up "
             "almost none of the training material. Short bars mean the accuracy "
             "held up;\n"
             f"for {better} of the {len(rows)} properties the less familiar "
             "boreholes were predicted slightly better.",
             fontsize=10, color=MUTED, ha="left", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_transfer_by_novelty.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_train_vs_heldout(
    results_path: Path | None = None,
    output_dir: Path | None = None,
    model: str = "unsupervised_vae",
    floor: float = -0.3,
) -> Path:
    """What the borehole holdout costs, property by property.

    The same regressor scored on the boreholes it was fitted on and on
    boreholes it never saw. This is the comparison the earlier draft
    made implicitly by reporting only the first number, and the length
    of each bar is what that choice was worth.

    Every property is shown, including the ones that fail. Presenting
    only the six that transfer would repeat, in the choice of rows, the
    same selection that made the original figures look strong. Values
    below `floor` are drawn at the floor with the true number printed,
    so a collapse is visible without one failure setting the scale for
    everything else.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    # One run carries both scores: `r2_train` is written unconditionally,
    # so this reads the same file the other holdout figures read.
    results_path = results_path or (RESULTS_DIR / "zeroshot_bootstrap.json")
    output_dir = Path(output_dir or PAPER_DIR)
    data = json.loads(Path(results_path).read_text())

    train, test = {}, {}
    for it in data["per_iteration"]:
        entry = next((m for m in it["models"] if m["model"] == model), None)
        for name, res in (entry or {}).get("per_variable", {}).items():
            if np.isfinite(res.get("r2_train", np.nan)):
                train.setdefault(name, []).append(res["r2_train"])
            if np.isfinite(res.get("r2", np.nan)):
                test.setdefault(name, []).append(res["r2"])

    rows = []
    for name in train:
        if name not in test:
            continue
        spec = TARGET_REGISTRY.get(name)
        label = (spec.label if spec else name).replace(
            "Magnetic susceptibility (point)", "Point magnetic susceptibility")
        if spec and spec.tier == "same_modality":
            label += "†"
        rows.append((label, float(np.median(train[name])), float(np.median(test[name]))))
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(rows) + 1.2))
    ys = np.arange(len(rows))

    for y, (_, a, b) in zip(ys, rows):
        ax.plot([max(b, floor), a], [y, y], color=GRID, lw=3.0,
                solid_capstyle="round", zorder=2)
    ax.scatter([r[1] for r in rows], ys, s=105, color=SERIES, zorder=4,
               linewidths=2, edgecolors="white")
    ax.scatter([max(r[2], floor) for r in rows], ys, s=105, color=REFERENCE,
               zorder=4, linewidths=2, edgecolors="white")

    for y, (_, a, b) in zip(ys, rows):
        ax.text(a + 0.028, y, f"{a:.2f}", ha="left", va="center",
                fontsize=9.5, color=INK)
        off = b < floor
        if off:
            # Printed to the right of the clipped dot; to its left it
            # collides with the property name.
            ax.text(floor + 0.030, y, f"◀ {b:.2f}", ha="left", va="center",
                    fontsize=9.5, color=ACCENT)
        else:
            ax.text(b - 0.028, y, f"{b:.2f}", ha="right", va="center",
                    fontsize=9.5, color=INK)

    ax.axvline(0.0, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(floor - 0.08, 1.06)
    ax.set_ylim(-0.7, len(rows) - 0.25)
    ax.set_xlabel("$R^2$", fontsize=10, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markersize=10, color=SERIES,
               label="in-sample"),
        Line2D([], [], marker="o", ls="", markersize=10, color=REFERENCE,
               label="held-out boreholes"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.06), ncols=2, frameon=False,
        fontsize=9.5, labelcolor=MUTED, handletextpad=0.4)

    # No title and no subtitle: this figure goes in a paper, where the caption
    # carries the reading, the dagger and the left-arrow marker.
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_train_vs_heldout.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_depth_within_between(
    within_pooled_heldout: list[tuple[str, float, float, float]],
    output_dir: Path | None = None,
) -> Path:
    """Depth explains a borehole and does not travel between boreholes.

    Three bars per property: depth fitted inside a single hole, depth
    fitted across pooled holes, and depth fitted on holes it has not
    seen. The collapse from left to right is the argument, and it is the
    reason the six measurements are worth having even though depth beats
    them comfortably within any one hole.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    output_dir = Path(output_dir or PAPER_DIR)
    rows = sorted(within_pooled_heldout, key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    series = [
        ("inside one borehole", [r[1] for r in rows], SEQUENTIAL[0]),
        ("boreholes pooled", [r[2] for r in rows], SEQUENTIAL[1]),
        ("a borehole it never saw", [r[3] for r in rows], SEQUENTIAL[2]),
    ]

    ys = np.arange(len(rows))
    height = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 0.62 * len(rows) + 2.8))

    for k, (name, values, colour) in enumerate(series):
        offset = (1 - k) * height
        ax.barh(ys + offset, values, height=height * 0.92, color=colour,
                edgecolor="white", linewidth=1.2, zorder=3, label=name)
        for y, v in zip(ys + offset, values):
            ax.text(v + 0.015 if v >= 0 else v - 0.015, y, f"{v:.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.2, color=INK)

    ax.axvline(0.0, color=MUTED, lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(-0.18, 1.12)
    ax.set_xlabel("$R^2$", fontsize=11, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), ncols=3,
              frameon=False, fontsize=9.5, labelcolor=MUTED, handlelength=1.2)

    # Title and reading live in the caption.
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "fig_depth_within_between.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
