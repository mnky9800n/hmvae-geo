"""How much of a property depth explains inside one borehole.

Section 4.5 reports depth fitted three ways: inside a single borehole,
pooled across boreholes, and on a borehole the fit has never seen. The
third comes from `depth_bootstrap`, which holds boreholes out. The first
two are in-sample by construction and are what this module measures.

That in-sample framing is deliberate rather than careless. The question
is what depth can explain at its best, because the argument is that even
at its best it does not travel: a coordinate that orders one hole almost
perfectly means something different in the next one. An out-of-sample
number for the within-hole case would answer a question nobody asked.

Two fits per target:

``within``   depth alone against the target, fitted and scored on the
             rows of a single borehole, then taken as the median over
             boreholes with enough rows to fit. An upper bound.
``pooled``   the same fit across every borehole at once, scored on the
             same rows. What depth explains when holes are mixed and
             their different depth-to-property relations average out.

The gap between them is the result: pooling destroys most of what depth
explains, before any holdout is imposed at all.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lilyvae.config import BIN_SIZE_CM, PROCESSED_DIR
from lilyvae.evaluation.depth_bootstrap import _fit_catboost, _prepare
from lilyvae.evaluation.targets import (
    HEADLINE_TIERS,
    TARGET_REGISTRY,
    select_targets,
)

#: A borehole needs at least this many valid rows before depth is fitted
#: inside it. Below this the in-sample R^2 is a curve through noise.
MIN_BOREHOLE_ROWS = 60


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def within_borehole_r2(
    depths: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    borehole_ids: np.ndarray,
) -> tuple[float, int]:
    """Median in-sample R^2 of depth against the target, hole by hole."""
    scores = []
    for hole in np.unique(borehole_ids[valid]):
        rows = valid & (borehole_ids == hole)
        n = int(rows.sum())
        if n < MIN_BOREHOLE_ROWS:
            continue
        y = values[rows]
        if np.ptp(y) == 0:
            continue
        x = depths[rows].reshape(-1, 1)
        model = _fit_catboost(x, y)
        score = _r2(y, model.predict(x))
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return float("nan"), 0
    return float(np.median(scores)), len(scores)


def pooled_r2(
    depths: np.ndarray, values: np.ndarray, valid: np.ndarray
) -> float:
    """In-sample R^2 of depth against the target, every borehole at once."""
    y = values[valid]
    if y.size < MIN_BOREHOLE_ROWS or np.ptp(y) == 0:
        return float("nan")
    x = depths[valid].reshape(-1, 1)
    model = _fit_catboost(x, y)
    return _r2(y, model.predict(x))


def run_depth_within(tiers: tuple[str, ...] = HEADLINE_TIERS) -> dict:
    """Measure the within-borehole and pooled depth fits for each target."""
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids = np.asarray(metadata["borehole_ids"])
    depths = np.asarray(metadata["depth_bins"], dtype=np.float64) * (
        BIN_SIZE_CM / 100.0
    )
    names = select_targets(targets, tiers)
    prepared = _prepare(targets, names)

    print(f"Depth within/pooled: {len(names)} targets, "
          f"{len(np.unique(borehole_ids))} boreholes")

    start = time.time()
    per_target = {}
    for name in names:
        values, valid, _ = prepared[name]
        within, n_holes = within_borehole_r2(depths, values, valid, borehole_ids)
        pooled = pooled_r2(depths, values, valid)
        per_target[name] = {
            "label": TARGET_REGISTRY[name].label,
            "within_borehole": within,
            "pooled": pooled,
            "n_boreholes_fitted": n_holes,
            "n_rows": int(valid.sum()),
        }
        print(f"  {TARGET_REGISTRY[name].label[:34]:<36}"
              f"within {within:+.3f} ({n_holes} holes)  pooled {pooled:+.3f}",
              flush=True)

    finite = [
        (v["within_borehole"], v["pooled"])
        for v in per_target.values()
        if np.isfinite(v["within_borehole"]) and np.isfinite(v["pooled"])
    ]
    summary = {
        "within_borehole_median": float(np.median([w for w, _ in finite])),
        "pooled_median": float(np.median([p for _, p in finite])),
        "n_targets": len(finite),
    }
    return {
        "config": {"tiers": list(tiers), "min_borehole_rows": MIN_BOREHOLE_ROWS},
        "per_target": per_target,
        "summary": summary,
        "elapsed_seconds": time.time() - start,
    }
