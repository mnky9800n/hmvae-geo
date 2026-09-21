"""Diagnose why zero-shot R^2 on SRM magnetic moment y is implausibly high.

Hypotheses:
  H1 near-constant column (R^2 of mean predictor ~= R^2 of any model)
  H2 dominant borehole / per-borehole leakage encoded in embeddings
  H3 unit or sign convention quirk vs x and z
  H4 outlier dominance (a few extreme samples carry the variance)

For each of SRM x / y / z we report:
  n, variance, per-borehole-variance share,
  R^2(embedding -> y), R^2(embedding -> y demeaned by borehole),
  R^2 with top 1% |y| removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import (
    BIN_SIZE_CM,
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)


def fit_predict_r2(X, y, seed=42):
    m = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.1,
        verbose=0, random_seed=seed,
        bootstrap_type="Bernoulli", subsample=0.8,
    )
    m.fit(X, y)
    return r2_score(y, m.predict(X))


def main():
    embeddings = torch.load(
        CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True
    ).numpy()
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    boreholes = np.asarray(metadata["borehole_ids"])
    depth_bins = np.asarray(metadata["depth_bins"])
    bin_m = BIN_SIZE_CM / 100.0

    main_df = pd.DataFrame({
        "_borehole": boreholes,
        "_depth_bin": depth_bins,
    })
    main_df["_idx"] = np.arange(len(main_df))

    print("Loading SRM_DataLITH.csv...")
    srm = pd.read_csv(RAW_DIR / "SRM_DataLITH.csv", low_memory=False)
    srm["Depth CSF-A (m)"] = pd.to_numeric(srm["Depth CSF-A (m)"], errors="coerce")
    srm = srm.dropna(subset=["Depth CSF-A (m)"])
    srm["_borehole"] = (
        srm["Exp"].astype(str) + "-"
        + srm["Site"].astype(str) + "-"
        + srm["Hole"].astype(str)
    )
    srm["_depth_bin"] = (srm["Depth CSF-A (m)"] / bin_m).round().astype(int)
    cols = [
        "Magnetic moment x (Am^2)",
        "Magnetic moment y (Am^2)",
        "Magnetic moment z (Am^2)",
    ]
    for c in cols:
        srm[c] = pd.to_numeric(srm[c], errors="coerce")

    binned = (
        srm.groupby(["_borehole", "_depth_bin"])[cols].mean().reset_index()
    )
    aligned = main_df.merge(binned, on=["_borehole", "_depth_bin"], how="left")

    print(f"\n{'axis':>6} {'n':>8} {'mean':>11} {'std':>11} {'min':>11} {'max':>11} {'within_BH%':>10}")
    print("-" * 80)

    for c, label in zip(cols, ["x", "y", "z"]):
        v = aligned[c].values
        mask = ~np.isnan(v)
        idx = aligned.loc[mask, "_idx"].values
        y_full = v[mask]
        bh = boreholes[idx]
        n = len(y_full)

        between_var = (
            pd.Series(y_full).groupby(bh).mean().var() if n else float("nan")
        )
        total_var = float(np.var(y_full)) if n else float("nan")
        within_share = (
            1.0 - between_var / total_var if total_var > 0 else float("nan")
        )

        print(
            f"{label:>6} {n:>8d} {y_full.mean():>11.3e} "
            f"{y_full.std():>11.3e} {y_full.min():>11.3e} {y_full.max():>11.3e} "
            f"{within_share*100:>9.1f}%"
        )

    print("\nCatBoost R^2 (embedding -> y) under several conditions:\n")
    print(f"{'axis':>6} {'n':>8} {'R2_full':>9} {'R2_BHdemean':>13} {'R2_no_top1pct':>15}")
    print("-" * 70)

    for c, label in zip(cols, ["x", "y", "z"]):
        v = aligned[c].values
        mask = ~np.isnan(v)
        idx = aligned.loc[mask, "_idx"].values
        y_full = v[mask]
        z_full = embeddings[idx]
        bh = boreholes[idx]
        if len(y_full) < 100:
            print(f"{label:>6} {len(y_full):>8d}  too few samples")
            continue

        r2_full = fit_predict_r2(z_full, y_full)

        bh_means = pd.Series(y_full).groupby(bh).transform("mean").values
        y_demean = y_full - bh_means
        if float(np.var(y_demean)) > 1e-20:
            r2_demean = fit_predict_r2(z_full, y_demean)
        else:
            r2_demean = float("nan")

        thr = np.quantile(np.abs(y_full), 0.99)
        keep = np.abs(y_full) <= thr
        r2_no_outlier = fit_predict_r2(z_full[keep], y_full[keep])

        print(
            f"{label:>6} {len(y_full):>8d} {r2_full:>9.4f} "
            f"{r2_demean:>13.4f} {r2_no_outlier:>15.4f}"
        )


if __name__ == "__main__":
    main()
