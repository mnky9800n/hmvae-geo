"""Bootstrap evaluation framework for zero-shot prediction."""

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import BOOTSTRAP_ITERATIONS, BOOTSTRAP_TEST_FRACTION, SEED


def bootstrap_zeroshot(
    embeddings: torch.Tensor,
    target_values: torch.Tensor,
    valid_mask: torch.Tensor,
    borehole_ids: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    test_fraction: float = BOOTSTRAP_TEST_FRACTION,
) -> dict:
    """Bootstrap evaluation of zero-shot prediction, splitting by borehole.

    For each iteration:
    1. Randomly partition boreholes into train/test
    2. Train CatBoost on training boreholes
    3. Evaluate R² on test boreholes

    Returns mean R² and 95% CI.
    """
    z = embeddings.numpy()
    vals = target_values.numpy()
    mask = valid_mask.numpy().astype(bool)

    # Only work with valid samples
    valid_indices = np.where(mask)[0]
    if len(valid_indices) < 50:
        return {"r2_mean": float("nan"), "r2_ci_low": float("nan"), "r2_ci_high": float("nan"), "n_valid": len(valid_indices)}

    z_valid = z[valid_indices]
    y_valid = vals[valid_indices]
    bh_valid = borehole_ids[valid_indices]

    unique_boreholes = np.unique(bh_valid)
    if len(unique_boreholes) < 5:
        return {"r2_mean": float("nan"), "r2_ci_low": float("nan"), "r2_ci_high": float("nan"), "n_valid": len(valid_indices)}

    rng = np.random.RandomState(SEED)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))
    r2_scores = []

    for i in range(n_iterations):
        perm = rng.permutation(unique_boreholes)
        test_boreholes = set(perm[:n_test])

        test_mask = np.array([bh in test_boreholes for bh in bh_valid])
        train_mask = ~test_mask

        if train_mask.sum() < 20 or test_mask.sum() < 10:
            continue

        model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0, random_seed=i, bootstrap_type="Bernoulli", subsample=0.8)
        model.fit(z_valid[train_mask], y_valid[train_mask])
        y_pred = model.predict(z_valid[test_mask])

        try:
            r2 = r2_score(y_valid[test_mask], y_pred)
            r2_scores.append(r2)
        except ValueError:
            continue

    if not r2_scores:
        return {"r2_mean": float("nan"), "r2_ci_low": float("nan"), "r2_ci_high": float("nan"), "n_valid": len(valid_indices)}

    r2_arr = np.array(r2_scores)
    return {
        "r2_mean": float(np.mean(r2_arr)),
        "r2_ci_low": float(np.percentile(r2_arr, 2.5)),
        "r2_ci_high": float(np.percentile(r2_arr, 97.5)),
        "n_iterations": len(r2_scores),
        "n_valid": len(valid_indices),
    }
