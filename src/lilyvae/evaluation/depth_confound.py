"""Depth-confound evaluation for zero-shot prediction.

Reviewer concern: zero-shot R^2 might be a depth-dependent compaction
signal rather than evidence that the embeddings encode rock properties
beyond depth. This module addresses that by comparing four CatBoost
zero-shot regressors per held-out variable:

    depth_only       y ~ CatBoost(depth)
    embedding_only   y ~ CatBoost(z)               [paper's headline]
    embedding_depth  y ~ CatBoost([z, depth])
    embedding_resid  (y - f(depth)) ~ CatBoost(z)  where f is GBM(depth)

If embedding_only is much higher than depth_only, the embeddings
contribute information beyond depth. embedding_resid is the cleanest
test: it asks whether embeddings predict the part of y that depth
cannot explain.

All evaluations are in-sample, matching the existing zero-shot protocol.
"""

from __future__ import annotations

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


def _fit_catboost(X: np.ndarray, y: np.ndarray, seed: int = 42) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_seed=seed,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(X, y)
    return model


def evaluate_depth_confound_variable(
    embeddings: torch.Tensor,
    depths: np.ndarray,
    target_values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Run the four CatBoost variants for a single held-out variable.

    Args:
        embeddings: (N, latent_dim) full-data embeddings.
        depths: (N,) depth in metres for every sample.
        target_values: (N,) values of the held-out variable. NaNs filled.
        valid_mask: (N,) bool mask of where target_values is real.

    Returns:
        dict with r2 values for each of the four regressors plus n_samples.
    """
    z = embeddings.numpy()
    d = depths.reshape(-1, 1).astype(np.float32)
    y = target_values.numpy()
    mask = valid_mask.numpy().astype(bool)

    z_v = z[mask]
    d_v = d[mask]
    y_v = y[mask]

    if len(y_v) < 50:
        return {
            "r2_depth_only": float("nan"),
            "r2_embedding_only": float("nan"),
            "r2_embedding_depth": float("nan"),
            "r2_embedding_resid": float("nan"),
            "n_samples": len(y_v),
        }

    m_depth = _fit_catboost(d_v, y_v)
    r2_depth = r2_score(y_v, m_depth.predict(d_v))

    m_emb = _fit_catboost(z_v, y_v)
    r2_emb = r2_score(y_v, m_emb.predict(z_v))

    zd = np.concatenate([z_v, d_v], axis=1)
    m_zd = _fit_catboost(zd, y_v)
    r2_zd = r2_score(y_v, m_zd.predict(zd))

    # Depth-residualised target: ask whether embeddings explain the
    # variance that depth alone cannot.
    resid = y_v - m_depth.predict(d_v)
    if float(np.var(resid)) < 1e-12:
        r2_resid = float("nan")
    else:
        m_resid = _fit_catboost(z_v, resid)
        r2_resid = r2_score(resid, m_resid.predict(z_v))

    return {
        "r2_depth_only": float(r2_depth),
        "r2_embedding_only": float(r2_emb),
        "r2_embedding_depth": float(r2_zd),
        "r2_embedding_resid": float(r2_resid),
        "n_samples": int(len(y_v)),
    }


def evaluate_all_depth_confound(
    embeddings: torch.Tensor,
    depths: np.ndarray,
    zeroshot_targets: dict,
) -> dict[str, dict]:
    """Run the four-way comparison for every held-out variable."""
    results: dict[str, dict] = {}
    for name, target in zeroshot_targets.items():
        print(f"  Depth-confound: {name} ({target['n_valid']} samples)...")
        try:
            r = evaluate_depth_confound_variable(
                embeddings,
                depths,
                target["values"],
                target["valid_mask"],
            )
            results[name] = r
            print(
                f"    depth={r['r2_depth_only']:.3f}  "
                f"emb={r['r2_embedding_only']:.3f}  "
                f"emb+depth={r['r2_embedding_depth']:.3f}  "
                f"emb|resid={r['r2_embedding_resid']:.3f}"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e), "n_samples": target["n_valid"]}
    return results
