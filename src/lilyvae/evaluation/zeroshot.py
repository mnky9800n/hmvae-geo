"""Zero-shot prediction using embeddings + CatBoost.

Targets are curated by `evaluation/targets.py` before they reach a
headline mean: procedural columns and algebraic duplicates are dropped,
and a few columns are winsorized because the LILY release contains
physically impossible outliers in them (see `TargetSpec.winsorize`).
Passing `tiers=None` restores the uncurated behaviour for comparison.
"""

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.evaluation.targets import (
    HEADLINE_TIERS,
    apply_valid_range,
    REPORTED_TIERS,
    TARGET_REGISTRY,
    select_targets,
    winsorize_values,
)


def evaluate_zeroshot_variable(
    embeddings: torch.Tensor,
    target_values: torch.Tensor,
    valid_mask: torch.Tensor,
    winsorize: bool = False,
) -> dict[str, float]:
    """Predict a single unseen variable from embeddings using CatBoost (in-sample).

    Args:
        embeddings: Per-sample latent vectors, row-aligned with the target.
        target_values: The held-out variable.
        valid_mask: True where the target was actually measured.
        winsorize: Clip both 1% tails before fitting. Set for targets whose
            LILY column contains corrupt extreme values; on a clean column
            it changes R^2 in the third decimal, on a corrupt one it is the
            difference between measuring the rock and memorising outliers.
    """
    z = embeddings.numpy()
    vals = target_values.numpy()
    mask = valid_mask.numpy().astype(bool)

    z_valid = z[mask]
    y_valid = vals[mask]

    if len(z_valid) < 50:
        return {"r2": float("nan"), "n_samples": len(z_valid)}

    n_clipped = 0
    if winsorize:
        y_valid, n_clipped = winsorize_values(y_valid)

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
        "n_winsorized": n_clipped,
    }


def evaluate_all_zeroshot(
    embeddings: torch.Tensor,
    zeroshot_targets: dict,
    tiers: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Run zero-shot prediction for the selected tiers of target variables.

    Args:
        embeddings: Per-sample latent vectors.
        zeroshot_targets: The dict saved by preprocessing.
        tiers: Curation tiers to evaluate. The default of None evaluates
            every harvested column and tags each result with its tier, so
            one run yields the curated headline, the same-modality group,
            and the uncurated number that the earlier draft reported.
            Use `summarize_zeroshot` to reduce the result to means.
    """
    if tiers is None:
        names = list(zeroshot_targets)
    else:
        names = select_targets(zeroshot_targets, tiers)

    results = {}
    for name in names:
        target = zeroshot_targets[name]
        spec = TARGET_REGISTRY.get(name)
        print(f"  Zero-shot: {name} ({target['n_valid']} samples)...")
        try:
            mask, n_dropped = apply_valid_range(
                target["values"].numpy(), target["valid_mask"].numpy().astype(bool),
                spec,
            )
            result = evaluate_zeroshot_variable(
                embeddings,
                target["values"],
                torch.from_numpy(mask),
                winsorize=bool(spec and spec.winsorize),
            )
            result["n_outside_valid_range"] = n_dropped
            if spec is not None:
                result["tier"] = spec.tier
                result["group"] = spec.group
            if n_dropped:
                print(f"    dropped {n_dropped} row(s) outside "
                      f"{spec.valid_range} as physically impossible")
            results[name] = result
            print(f"    R² = {result['r2']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"r2": float("nan"), "n_samples": target["n_valid"], "error": str(e)}
    return results


def summarize_zeroshot(results: dict[str, dict]) -> dict:
    """Reduce per-variable zero-shot results to one mean per curation tier.

    The headline number is the mean over the `independent` tier: physical
    properties that are neither a training variable measured by another
    instrument nor an algebraic restatement of another target. The
    uncurated mean over every harvested column is reported alongside it,
    because the difference between the two is itself a result about how
    much of an apparent zero-shot score comes from counting one
    measurement several times.
    """
    by_tier: dict[str, list[float]] = {}
    for name, res in results.items():
        r2 = res.get("r2")
        if r2 is None or not np.isfinite(r2):
            continue
        tier = res.get("tier") or (
            TARGET_REGISTRY[name].tier if name in TARGET_REGISTRY else "unknown"
        )
        by_tier.setdefault(tier, []).append(float(r2))

    def _mean(tiers: tuple[str, ...]) -> tuple[float, int]:
        vals = [v for t in tiers for v in by_tier.get(t, [])]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    headline, n_headline = _mean(HEADLINE_TIERS)
    reported, n_reported = _mean(REPORTED_TIERS)
    everything, n_all = _mean(tuple(by_tier))

    return {
        "r2_headline": headline,
        "n_headline": n_headline,
        "r2_reported": reported,
        "n_reported": n_reported,
        "r2_uncurated": everything,
        "n_uncurated": n_all,
        "per_tier": {
            t: {"r2_mean": float(np.mean(v)), "n": len(v)}
            for t, v in sorted(by_tier.items())
        },
    }
