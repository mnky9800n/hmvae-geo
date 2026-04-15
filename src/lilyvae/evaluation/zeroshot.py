"""Zero-shot prediction using embeddings + CatBoost."""

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


def evaluate_zeroshot_variable(
    embeddings: torch.Tensor,
    target_values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Predict a single unseen variable from embeddings using CatBoost (in-sample)."""
    z = embeddings.numpy()
    vals = target_values.numpy()
    mask = valid_mask.numpy().astype(bool)

    z_valid = z[mask]
    y_valid = vals[mask]

    if len(z_valid) < 50:
        return {"r2": float("nan"), "n_samples": len(z_valid)}

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_seed=42,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(z_valid, y_valid)
    y_pred = model.predict(z_valid)
    r2 = r2_score(y_valid, y_pred)

    return {
        "r2": float(r2),
        "n_samples": len(z_valid),
    }


def evaluate_all_zeroshot(
    embeddings: torch.Tensor,
    zeroshot_targets: dict,
) -> dict[str, dict]:
    """Run zero-shot prediction for all available target variables."""
    results = {}
    for name, target in zeroshot_targets.items():
        print(f"  Zero-shot: {name} ({target['n_valid']} samples)...")
        try:
            result = evaluate_zeroshot_variable(
                embeddings,
                target["values"],
                target["valid_mask"],
            )
            results[name] = result
            print(f"    R² = {result['r2']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"r2": float("nan"), "n_samples": target["n_valid"], "error": str(e)}
    return results
