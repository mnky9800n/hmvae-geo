"""Figure colours, in one place.

The categorical pair is the teal and purple of the Solo Jazz cup, which
is a request but not an indulgence: the two hues sit far apart in both
hue and lightness, so they survive the checks a scientific figure has to
pass. Validated against a white surface with the data-visualisation
palette validator:

    lightness band      both inside L 0.43-0.77
    chroma floor        both >= 0.1
    CVD separation      dE 21.9 deutan, 28.5 tritan
    normal vision       dE 30.7
    contrast vs white   both >= 3:1

Anything replacing them has to clear the same bars, and the adjacent
pair is what matters, not the individual colours.

Where a comparison is ordered rather than categorical, use SEQUENTIAL
instead. Within-borehole, pooled and held-out is a sequence of
increasing demand, not three unrelated things, so it takes one hue
darkening rather than three hues.
"""

from __future__ import annotations

#: Categorical slots. Two is deliberate: no figure in this paper needs a
#: third unordered series, and inventing one would mean inventing a hue.
TEAL = "#009EA0"
PURPLE = "#6E2B8B"
CATEGORICAL = (TEAL, PURPLE)

#: One hue, light to dark, for ordered comparisons.
SEQUENTIAL = ("#7FD4D5", "#009EA0", "#005F60")

#: Text and furniture. These are ink, never series colour: a value or an
#: axis label wearing a series colour makes the reader decode twice.
INK = "#141414"
MUTED = "#55555b"
GRID = "#dcdcde"
SURFACE = "#ffffff"

#: Diverging, for a signed quantity. Two hues with a neutral midpoint,
#: never a hue at zero: a coloured midpoint invents a category where the
#: data has none. Teal for negative, purple for positive.
DIVERGING_LOW = TEAL
DIVERGING_MID = "#f4f4f2"
DIVERGING_HIGH = PURPLE


def diverging_cmap(n_bands: int = 20):
    """Banded diverging colormap and its norm.

    Banded rather than continuous because a reader cannot recover a value
    from a smooth ramp: the bands let a cell be matched to a number on
    the bar, which is what a correlation matrix is read for.
    """
    import numpy as np
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "teal_white_purple",
        [DIVERGING_LOW, "#7FD4D5", DIVERGING_MID, "#B69ACB", DIVERGING_HIGH],
    ).resampled(n_bands)
    bounds = np.linspace(-1.0, 1.0, n_bands + 1)
    return cmap, BoundaryNorm(bounds, cmap.N), bounds


#: Drawn when a value is clipped or a fit is not distinguishable from
#: flat, so the exception is visible without adding a hue.
ACCENT = "#c2185b"


def apply_palette() -> None:
    """Set the matplotlib defaults this palette implies."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "axes.edgecolor": GRID,
    })
