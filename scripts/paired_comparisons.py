"""Compare models on the folds they share, rather than on their intervals.

Every model in the bootstrap is trained and scored on the same 30
borehole splits, so the comparison is paired and a fold that is hard for
one model is hard for all of them. Reading two marginal interquartile
ranges and noting that they overlap throws that pairing away and is far
less sensitive: the unsupervised model beats principal components by
0.030 adjusted Rand index at p = 0.0007, while their interquartile
ranges overlap across most of their width.

It also stops differences being reported that are not there. A
difference inside the spread is not a difference, and calling one model
nominally ahead of another still plants an ordering the data does not
support.

    uv run python scripts/paired_comparisons.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from lilyvae.config import RESULTS_DIR

REFERENCE = "unsupervised_vae"
METRICS = ("ari", "auc_macro", "r2_avg")
ALPHA = 0.05


def per_fold(results: dict) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for iteration in results["per_iteration"]:
        for model in iteration["models"]:
            if "error" in model:
                continue
            store = out.setdefault(model["model"], {})
            for metric in METRICS:
                value = model.get(metric)
                if value is not None and np.isfinite(value):
                    store.setdefault(metric, []).append(float(value))
    return out


def compare(a: list[float], b: list[float]) -> tuple[float, float, int]:
    n = min(len(a), len(b))
    x, y = np.array(a[:n]), np.array(b[:n])
    if n < 5 or np.allclose(x, y):
        return float(np.median(x - y)), float("nan"), n
    return float(np.median(x - y)), float(wilcoxon(x, y).pvalue), n


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        RESULTS_DIR / "bootstrap_all_models.json")
    folds = per_fold(json.loads(path.read_text()))
    if REFERENCE not in folds:
        print(f"{REFERENCE} not in {path}")
        return 1

    rows = []
    print(f"Paired against {REFERENCE} on shared folds (Wilcoxon signed-rank)\n")
    print(f"{'model':<26} {'metric':<10} {'median diff':>12} {'p':>9}  verdict")
    print("-" * 78)
    for name in sorted(folds):
        if name == REFERENCE:
            continue
        for metric in METRICS:
            if metric not in folds[name] or metric not in folds[REFERENCE]:
                continue
            diff, p, n = compare(folds[REFERENCE][metric], folds[name][metric])
            resolvable = np.isfinite(p) and p < ALPHA
            verdict = "resolvable" if resolvable else "no resolvable difference"
            rows.append({"model": name, "metric": metric, "median_difference": diff,
                         "p_value": p, "n_folds": n, "resolvable": bool(resolvable)})
            print(f"{name:<26} {metric:<10} {diff:>+12.4f} {p:>9.4f}  {verdict}")

    out = RESULTS_DIR / "paired_comparisons.json"
    out.write_text(json.dumps(
        {"reference": REFERENCE, "alpha": ALPHA, "comparisons": rows},
        indent=2, default=float))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
