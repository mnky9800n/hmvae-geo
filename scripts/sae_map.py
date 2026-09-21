"""One-off: train an SAE on VAE latents, match each feature to one of four
geological hypotheses, and plot the hypothesis-best feature on a global map.

Hypotheses (defined in `lilyvae.evaluation.sae`):
1. depth / compaction        -- |Pearson r| with depth
2. carbonate vs clay         -- carbonate fraction minus clay fraction at top-k
3. volcanic / tephra         -- volcanic-keyword fraction at top-k
4. organic-rich              -- Pearson r with NGR

For comparison the script also prints (but does not plot) the top features by
cross-borehole variance, which is what an "outlier-borehole" ranking selects.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from lilyvae.config import CHECKPOINT_DIR, PROCESSED_DIR, RESULTS_DIR
from lilyvae.evaluation.sae import (
    SAETrainConfig,
    find_best_matches,
    score_features_against_targets,
    train_sae,
)
from lilyvae.plotting.style import apply_style


HYPOTHESIS_TITLES = {
    "depth": "Depth / compaction",
    "carbonate_vs_clay": "Carbonate vs clay",
    "volcanic": "Volcanic / tephra",
    "organic_rich": "Organic-rich (high NGR)",
}


def _hypothesis_score(name: str, score) -> tuple[str, float]:
    """Return a short label and the numerical value used to select this feature."""
    if name == "depth":
        return "|depth r|", abs(score.depth_corr)
    if name == "carbonate_vs_clay":
        return "carb-clay", score.carbonate_vs_clay
    if name == "volcanic":
        return "volcanic frac", score.volcanic_frac
    if name == "organic_rich":
        return "NGR r", score.ngr_corr
    return "score", 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--l1-coeff", type=float, default=3e-3)
    parser.add_argument("--top-k", type=int, default=500,
                        help="top activating samples for lithology fraction stats")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "sae_map.png")
    parser.add_argument("--embeddings", type=Path,
                        default=CHECKPOINT_DIR / "unsupervised_embeddings.pt")
    args = parser.parse_args()

    apply_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading latents from {args.embeddings}")
    z = torch.load(args.embeddings, weights_only=False).float()
    md = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    train_features = torch.load(PROCESSED_DIR / "train_features.pt",
                                weights_only=False)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt",
                        weights_only=False).numpy()
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    label_classes = list(le.classes_)

    feature_columns = md["feature_columns"]
    ngr_col = feature_columns.index("NGR")
    ngr = train_features[:, ngr_col].numpy().astype(np.float64)

    lat = np.asarray(md["latitude"], dtype=np.float64)
    lon = np.asarray(md["longitude"], dtype=np.float64)
    depth_bin = np.asarray(md["depth_bins"], dtype=np.float64)
    borehole = np.asarray(md["borehole_ids"])

    valid = np.isfinite(lat) & np.isfinite(lon)
    print(f"Samples: {len(z)}, with lat/lon: {valid.sum()}")
    z = z[valid]
    lat = lat[valid]
    lon = lon[valid]
    depth_bin = depth_bin[valid]
    borehole = borehole[valid]
    labels = labels[valid]
    ngr = ngr[valid]

    print(f"Training SAE (d_hidden={args.d_hidden}, epochs={args.num_epochs})")
    cfg = SAETrainConfig(
        d_hidden=args.d_hidden,
        num_epochs=args.num_epochs,
        l1_coeff=args.l1_coeff,
    )
    sae, diag = train_sae(z, cfg)
    print(f"  recon R^2={diag['recon_r2']:.3f}  mean L0={diag['mean_l0']:.2f}")

    z_mean = diag["z_mean"]
    z_std = diag["z_std"]
    z_norm = (z - z_mean) / z_std
    with torch.no_grad():
        feats = sae.encode(z_norm).numpy()  # (N, d_hidden)

    # Score every feature against the four target hypotheses.
    scores = score_features_against_targets(
        sae=sae,
        z=z,
        labels=labels,
        depth=depth_bin,
        ngr=ngr,
        label_classes=label_classes,
        z_mean=z_mean,
        z_std=z_std,
        top_k=args.top_k,
    )
    best = find_best_matches(scores)

    # Distribution of activations across all features (for the user's question
    # "are most features close to zero?").
    act_freq_all = (feats > 0).mean(0)
    mean_act_all = feats.mean(0)
    print("\nActivation distribution across all features:")
    print(f"  median activation frequency: {np.median(act_freq_all):.3f}")
    print(f"  median mean activation:      {np.median(mean_act_all):.3f}")
    print(f"  features with freq < 0.05:   "
          f"{int((act_freq_all < 0.05).sum())} / {len(act_freq_all)}")

    # Per-borehole means for plotting.
    df = pd.DataFrame({"borehole": borehole, "lat": lat, "lon": lon})
    per_bh = df.groupby("borehole").agg(
        lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"),
    ).reset_index()
    bh_index = pd.Categorical(borehole, categories=per_bh["borehole"].values)
    bh_codes = bh_index.codes
    n_bh = len(per_bh)
    d_hidden = feats.shape[1]
    sums = np.zeros((n_bh, d_hidden), dtype=np.float64)
    counts = np.zeros(n_bh, dtype=np.float64)
    np.add.at(sums, bh_codes, feats)
    np.add.at(counts, bh_codes, 1)
    bh_mean = sums / np.maximum(counts, 1)[:, None]
    bh_var = bh_mean.var(axis=0)

    print("\nHypothesis-best features:")
    print(f"  {'hypothesis':<22}  {'feat':>4}  {'score':>10}  "
          f"{'depth r':>8}  {'NGR r':>7}  {'cross-bh var':>12}  {'act freq':>9}")
    for name in ("depth", "carbonate_vs_clay", "volcanic", "organic_rich"):
        s = best[name]
        score_label, score_val = _hypothesis_score(name, s)
        print(f"  {name:<22}  {s.feature_idx:>4d}  "
              f"{score_val:>+10.3f}  {s.depth_corr:>+8.3f}  "
              f"{s.ngr_corr:>+7.3f}  {bh_var[s.feature_idx]:>12.4f}  "
              f"{s.activation_frequency:>9.3f}")

    # Sanity-check: print the top features by cross-borehole variance for
    # comparison with the hypothesis-driven selection.
    top_var = np.argsort(-bh_var)[:6]
    print("\nFor comparison, top features by cross-borehole variance:")
    print(f"  {'feat':>4}  {'cross-bh var':>12}  {'depth r':>8}  "
          f"{'NGR r':>7}  {'act freq':>9}")
    for k in top_var:
        s = scores[k]
        print(f"  {k:>4d}  {bh_var[k]:>12.4f}  {s.depth_corr:>+8.3f}  "
              f"{s.ngr_corr:>+7.3f}  {s.activation_frequency:>9.3f}")

    # --- Plot: 2x2 grid, one panel per hypothesis ---
    fig = plt.figure(figsize=(14, 7))
    proj = ccrs.Robinson(central_longitude=103.8)  # Singapore

    for i, name in enumerate(("depth", "carbonate_vs_clay", "volcanic", "organic_rich")):
        s = best[name]
        k = s.feature_idx
        score_label, score_val = _hypothesis_score(name, s)

        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#eeeeee", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#888888")

        vals = bh_mean[:, k]
        vmax = np.quantile(vals, 0.98) if np.any(vals > 0) else 1.0
        sc = ax.scatter(
            per_bh["lon"], per_bh["lat"],
            c=vals, cmap="viridis", vmin=0, vmax=vmax,
            s=12 + 2 * np.log1p(per_bh["n"]),
            transform=ccrs.PlateCarree(),
            edgecolor="black", linewidth=0.3,
        )
        title = (
            f"{HYPOTHESIS_TITLES[name]}: feature {k}\n"
            f"{score_label}={score_val:+.2f}, depth r={s.depth_corr:+.2f}, "
            f"act freq={s.activation_frequency:.2f}"
        )
        ax.set_title(title, fontsize=10)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("mean activation", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("SAE features matched to geological hypotheses", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
