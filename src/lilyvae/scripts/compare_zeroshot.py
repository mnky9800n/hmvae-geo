"""Compare zero-shot prediction across all models with borehole-level bootstrap.

Models compared:
  - raw_features: input variables used directly as the embedding
  - pca:          PCA projection (fit on training boreholes per iteration)
  - unsupervised_vae:   pre-trained unsupervised VAE embeddings
  - semi_supervised_vae: pre-trained semi-supervised VAE embeddings

Neural autoencoder baselines (LinearAutoencoder, DeterministicAutoencoder) are
intentionally excluded.  Their embeddings are produced by a model trained on the
full dataset, so splitting boreholes after the fact would leak test-borehole data
into the encoder.  Including them would require re-training inside every bootstrap
iteration, which is not equivalent to the zero-shot setting used for the VAE models.

Bootstrap methodology:
  For each iteration the same borehole partition is applied to every model, so all
  R² values are directly comparable.  Splits are generated once, before the model
  loop, from a single seeded RNG.
"""

import argparse
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SEED,
    LATENT_DIM,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_TEST_FRACTION,
)

# Minimum number of valid samples required to include a target variable.
# Mirrors the guard used in bootstrap.bootstrap_zeroshot.
_MIN_VALID_SAMPLES = 50
_MIN_UNIQUE_BOREHOLES = 5


def _generate_borehole_splits(
    borehole_ids: np.ndarray,
    n_iterations: int,
    test_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate borehole-level train/test index masks for all iterations.

    Generating splits once before the model loop is the only way to ensure that
    every model is evaluated on identical partitions.  An independent RNG per
    model would produce the same sequences only so long as the loop body never
    calls the RNG a different number of times — a fragile assumption.

    Returns:
        List of (train_mask, test_mask) pairs, one per iteration, where each
        mask is a boolean array over all samples.
    """
    unique_boreholes = np.unique(borehole_ids)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))
    rng = np.random.RandomState(SEED)

    splits = []
    for _ in range(n_iterations):
        perm = rng.permutation(unique_boreholes)
        test_bh = set(perm[:n_test])
        test_mask = np.array([bh in test_bh for bh in borehole_ids])
        splits.append((~test_mask, test_mask))

    return splits


def _select_targets(zeroshot_targets: dict) -> dict:
    """Return the subset of targets that have enough global valid samples.

    This is evaluated once against the full dataset.  The per-iteration
    minimum-sample guard inside _bootstrap_with_splits provides a second
    check at the split level.
    """
    return {
        name: t
        for name, t in zeroshot_targets.items()
        if int(t["valid_mask"].numpy().astype(bool).sum()) >= _MIN_VALID_SAMPLES
    }


def _bootstrap_with_splits(
    embeddings: np.ndarray,
    targets: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate zero-shot R² using pre-generated borehole splits.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).  For PCA these
            are the full-dataset projections; the caller is responsible for
            re-fitting PCA per split when leakage matters.
        targets: Dict mapping variable name to {"values": Tensor,
            "valid_mask": Tensor} — the same structure as zeroshot_targets.pt.
        splits: Pre-generated (train_mask, test_mask) pairs.  All models in a
            comparison run must receive the same list.

    Returns:
        Dict with r2_mean, r2_ci_low, r2_ci_high, median, iqr_low, iqr_high,
        n_iterations — consistent with bootstrap.bootstrap_zeroshot's schema.
    """
    per_iteration_r2 = []

    for iteration_idx, (train_mask, test_mask) in enumerate(splits):
        iter_r2s = []

        for target in targets.values():
            valid = target["valid_mask"].numpy().astype(bool)
            vals = target["values"].numpy()

            train_idx = train_mask & valid
            test_idx = test_mask & valid

            if train_idx.sum() < 20 or test_idx.sum() < 10:
                continue

            model = CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_seed=iteration_idx,
                bootstrap_type="Bernoulli",
                subsample=0.8,
            )
            model.fit(embeddings[train_idx], vals[train_idx])
            pred = model.predict(embeddings[test_idx])

            try:
                r2 = r2_score(vals[test_idx], pred)
                iter_r2s.append(r2)
            except ValueError:
                # r2_score raises ValueError when the test set has zero variance.
                continue

        if iter_r2s:
            per_iteration_r2.append(float(np.mean(iter_r2s)))

    if not per_iteration_r2:
        nan = float("nan")
        return {
            "r2_mean": nan,
            "r2_ci_low": nan,
            "r2_ci_high": nan,
            "median": nan,
            "iqr_low": nan,
            "iqr_high": nan,
            "n_iterations": 0,
        }

    arr = np.array(per_iteration_r2)
    return {
        "r2_mean": float(np.mean(arr)),
        "r2_ci_low": float(np.percentile(arr, 2.5)),
        "r2_ci_high": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "iqr_low": float(np.percentile(arr, 25)),
        "iqr_high": float(np.percentile(arr, 75)),
        "n_iterations": len(per_iteration_r2),
    }


def _bootstrap_pca_with_splits(
    X: np.ndarray,
    targets: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
    latent_dim: int,
) -> dict:
    """Evaluate PCA zero-shot R² with PCA fit restricted to training boreholes.

    PCA must be fit only on training-split samples.  Fitting on the full dataset
    and then splitting would leak test-borehole variance structure into the
    projection, which the VAE baselines do not benefit from.
    """
    per_iteration_r2 = []

    for iteration_idx, (train_mask, test_mask) in enumerate(splits):
        # Fit PCA on training boreholes only.
        n_components = min(latent_dim, X.shape[1], train_mask.sum())
        pca = PCA(n_components=n_components, random_state=SEED)
        pca.fit(X[train_mask])
        embeddings = pca.transform(X)

        iter_r2s = []
        for target in targets.values():
            valid = target["valid_mask"].numpy().astype(bool)
            vals = target["values"].numpy()

            train_idx = train_mask & valid
            test_idx = test_mask & valid

            if train_idx.sum() < 20 or test_idx.sum() < 10:
                continue

            model = CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_seed=iteration_idx,
                bootstrap_type="Bernoulli",
                subsample=0.8,
            )
            model.fit(embeddings[train_idx], vals[train_idx])
            pred = model.predict(embeddings[test_idx])

            try:
                r2 = r2_score(vals[test_idx], pred)
                iter_r2s.append(r2)
            except ValueError:
                continue

        if iter_r2s:
            per_iteration_r2.append(float(np.mean(iter_r2s)))

    if not per_iteration_r2:
        nan = float("nan")
        return {
            "r2_mean": nan,
            "r2_ci_low": nan,
            "r2_ci_high": nan,
            "median": nan,
            "iqr_low": nan,
            "iqr_high": nan,
            "n_iterations": 0,
        }

    arr = np.array(per_iteration_r2)
    return {
        "r2_mean": float(np.mean(arr)),
        "r2_ci_low": float(np.percentile(arr, 2.5)),
        "r2_ci_high": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "iqr_low": float(np.percentile(arr, 25)),
        "iqr_high": float(np.percentile(arr, 75)),
        "n_iterations": len(per_iteration_r2),
    }


def _print_result(name: str, r: dict) -> None:
    print(
        f"  {name:30s}  mean={r['r2_mean']:.4f}"
        f"  median={r['median']:.4f}"
        f"  IQR=[{r['iqr_low']:.4f}, {r['iqr_high']:.4f}]"
        f"  n={r['n_iterations']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare zero-shot R² across all models (out-of-sample, borehole bootstrap)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
        help="Number of bootstrap iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=BOOTSTRAP_TEST_FRACTION,
        help="Fraction of boreholes held out for testing (default: %(default)s)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids: np.ndarray = metadata["borehole_ids"]
    X = features.numpy()

    # ------------------------------------------------------------------
    # Build shared splits and filter targets once.
    # Every model receives the same splits — this is what makes the R²
    # values directly comparable.
    # ------------------------------------------------------------------
    splits = _generate_borehole_splits(borehole_ids, args.iterations, args.test_fraction)
    targets = _select_targets(zeroshot_targets)

    if not targets:
        print(f"No target variables have >= {_MIN_VALID_SAMPLES} valid samples. Exiting.")
        return

    print(f"Evaluating {len(targets)} target variable(s) over {len(splits)} bootstrap iterations.")
    print(f"Borehole test fraction: {args.test_fraction:.0%}\n")

    results = {}

    # ------------------------------------------------------------------
    # 1. PCA — fit on training boreholes only per iteration
    # ------------------------------------------------------------------
    print("PCA...")
    results["pca"] = _bootstrap_pca_with_splits(X, targets, splits, LATENT_DIM)
    _print_result("pca", results["pca"])

    # ------------------------------------------------------------------
    # 2. Raw features — no embedding transformation
    # ------------------------------------------------------------------
    print("Raw features...")
    results["raw_features"] = _bootstrap_with_splits(X, targets, splits)
    _print_result("raw_features", results["raw_features"])

    # ------------------------------------------------------------------
    # 3. Unsupervised VAE
    # ------------------------------------------------------------------
    unsup_path = CHECKPOINT_DIR / "unsupervised_embeddings.pt"
    if unsup_path.exists():
        print("Unsupervised VAE...")
        unsup_emb = torch.load(unsup_path, weights_only=True).numpy()
        results["unsupervised_vae"] = _bootstrap_with_splits(unsup_emb, targets, splits)
        _print_result("unsupervised_vae", results["unsupervised_vae"])
    else:
        print(f"Unsupervised VAE embeddings not found at {unsup_path}, skipping.")

    # ------------------------------------------------------------------
    # 4. Semi-supervised VAE
    # ------------------------------------------------------------------
    semi_path = CHECKPOINT_DIR / "semi_supervised_embeddings.pt"
    if semi_path.exists():
        print("Semi-supervised VAE...")
        semi_emb = torch.load(semi_path, weights_only=True).numpy()
        results["semi_supervised_vae"] = _bootstrap_with_splits(semi_emb, targets, splits)
        _print_result("semi_supervised_vae", results["semi_supervised_vae"])
    else:
        print(f"Semi-supervised VAE embeddings not found at {semi_path}, skipping.")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Out-of-sample zero-shot comparison (borehole bootstrap)")
    print("=" * 70)
    header = f"  {'model':30s}  {'mean':>8}  {'median':>8}  {'IQR':>20}  {'n':>4}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        iqr = f"[{r['iqr_low']:.4f}, {r['iqr_high']:.4f}]"
        print(
            f"  {name:30s}  {r['r2_mean']:8.4f}  {r['median']:8.4f}"
            f"  {iqr:>20}  {r['n_iterations']:>4}"
        )

    out_path = RESULTS_DIR / "zeroshot_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
