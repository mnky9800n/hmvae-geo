"""Monte Carlo exploration of what maximizes zero-shot prediction."""

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import SEED


def monte_carlo_explore(
    embeddings: torch.Tensor,
    zeroshot_targets: dict,
    borehole_ids: np.ndarray,
    n_trials: int = 500,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """Explore which configurations maximize zero-shot prediction R².

    For each trial, randomly samples:
    - A subset of latent dimensions (1 to all)
    - CatBoost hyperparameters (depth, learning_rate, iterations)

    For each target variable, evaluates on a borehole-level train/test split.

    Returns DataFrame with all trial configurations and R² scores.
    """
    rng = np.random.RandomState(SEED)
    z = embeddings.numpy()
    latent_dim = z.shape[1]
    unique_boreholes = np.unique(borehole_ids)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))

    records = []

    for trial in range(n_trials):
        # Random configuration
        n_dims = rng.randint(1, latent_dim + 1)
        dim_subset = sorted(rng.choice(latent_dim, size=n_dims, replace=False).tolist())
        depth = rng.choice([4, 6, 8, 10])
        lr = rng.choice([0.01, 0.05, 0.1, 0.2])
        iterations = rng.choice([200, 500, 1000])

        # Random borehole split
        perm = rng.permutation(unique_boreholes)
        test_boreholes = set(perm[:n_test])

        z_subset = z[:, dim_subset]

        for target_name, target in zeroshot_targets.items():
            vals = target["values"].numpy()
            mask = target["valid_mask"].numpy().astype(bool)
            valid_idx = np.where(mask)[0]
            if len(valid_idx) < 50:
                continue

            z_v = z_subset[valid_idx]
            y_v = vals[valid_idx]
            bh_v = borehole_ids[valid_idx]

            test_mask = np.array([bh in test_boreholes for bh in bh_v])
            train_mask = ~test_mask

            if train_mask.sum() < 20 or test_mask.sum() < 10:
                continue

            try:
                model = CatBoostRegressor(
                    iterations=int(iterations),
                    depth=int(depth),
                    learning_rate=float(lr),
                    verbose=0,
                    random_seed=trial,
                )
                model.fit(z_v[train_mask], y_v[train_mask])
                y_pred = model.predict(z_v[test_mask])
                r2 = r2_score(y_v[test_mask], y_pred)
            except Exception:
                r2 = float("nan")

            records.append({
                "trial": trial,
                "target": target_name,
                "n_dims": n_dims,
                "dims": str(dim_subset),
                "catboost_depth": int(depth),
                "catboost_lr": float(lr),
                "catboost_iterations": int(iterations),
                "r2": r2,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            })

        if (trial + 1) % 50 == 0:
            print(f"  MC trial {trial+1}/{n_trials}")

    return pd.DataFrame(records)
