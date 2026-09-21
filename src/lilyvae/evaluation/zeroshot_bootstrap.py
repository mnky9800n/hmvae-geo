"""Zero-shot prediction evaluated on boreholes the model never saw.

Why this is separate from `evaluation/zeroshot.py`
--------------------------------------------------
That module fits CatBoost on the embeddings and scores it on the same
rows. That is a real quantity -- it asks whether a held-out variable is
*recoverable* from the representation at all -- but it is not
generalisation, and the earlier draft reported it as though it were.

The obvious repair, a single held-out-borehole split, trades one problem
for another: boreholes are geologically non-exchangeable, so any single
split is biased by whichever boreholes happen to land on the held-out
side. A fold containing the only borehole with a given lithology will
score badly for reasons that have nothing to do with the model.

So neither number is reported alone. This module measures **transfer**:
the VAE is retrained on the training boreholes of each fold, CatBoost is
fitted on those boreholes' rows only, and both are scored on boreholes
absent from both fits. Across the same folds used by
`evaluation/bootstrap.py`, the median says how well the representation
carries and the interquartile range says how much borehole-to-borehole
heterogeneity costs. The spread *is* the measurement of
non-exchangeability rather than a defect concealed by it.

Each fold also records how far its held-out boreholes sit from the
training set (lithology Jaccard overlap, energy distance in the six
input features), so the question "did this fold score badly because the
model is weak or because the fold was hard?" is answerable from the
output rather than by assertion.

Blocked cross-validation of this kind is standard where samples are
spatially structured: see Roberts et al. (2017) and Ploton et al.
(2020).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import PROCESSED_DIR, SEED
from lilyvae.evaluation.bootstrap import ALL_TRAINERS, MODEL_REGISTRY
from lilyvae.evaluation.folds import borehole_splits, energy_distance, lithology_jaccard
from lilyvae.evaluation.targets import (
    HEADLINE_TIERS,
    apply_valid_range,
    REPORTED_TIERS,
    TARGET_REGISTRY,
    select_targets,
    winsorize_values,
)
from lilyvae.training.trainer import get_device

#: Minimum rows on each side of a fold before a target is scored. Below
#: this the R^2 is too noisy to mean anything and is recorded as nan.
MIN_TRAIN_ROWS = 50
MIN_TEST_ROWS = 25


def _fit_catboost(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> CatBoostRegressor:
    """Fit the same regressor `evaluation/zeroshot.py` uses.

    Hyperparameters are deliberately identical so that the difference
    between recoverability and transfer is a difference of protocol
    alone.
    """
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_seed=seed,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(x, y)
    return model


def evaluate_zeroshot_transfer(
    embeddings: np.ndarray,
    target_values: np.ndarray,
    valid_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    winsorize: bool = False,
) -> dict:
    """Fit on training-borehole rows, score on held-out-borehole rows.

    Args:
        embeddings: Latents for every row, produced by a VAE that saw
            only the training boreholes.
        target_values: The held-out variable, full length.
        valid_mask: True where the variable was actually measured.
        train_mask: True for rows in this fold's training boreholes.
        test_mask: True for rows in this fold's held-out boreholes.
        winsorize: Clip both 1% tails, using thresholds taken from the
            training rows only so the held-out side leaks nothing.

    Returns:
        dict with r2, n_train, n_test, and n_winsorized.
    """
    fit_rows = valid_mask & train_mask
    score_rows = valid_mask & test_mask
    n_train, n_test = int(fit_rows.sum()), int(score_rows.sum())

    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        return {"r2": float("nan"), "n_train": n_train, "n_test": n_test}

    y_train = np.asarray(target_values, dtype=np.float64)[fit_rows]
    y_test = np.asarray(target_values, dtype=np.float64)[score_rows]

    n_clipped = 0
    if winsorize:
        _, n_clipped = winsorize_values(y_train)
        lo, hi = np.percentile(y_train, [1.0, 99.0])
        y_train = np.clip(y_train, lo, hi)
        y_test = np.clip(y_test, lo, hi)

    # A held-out slice can be constant (one borehole, one instrument
    # setting); R^2 is undefined there rather than zero.
    if np.ptp(y_test) == 0:
        return {
            "r2": float("nan"),
            "n_train": n_train,
            "n_test": n_test,
            "n_winsorized": n_clipped,
            "note": "held-out target is constant",
        }

    model = _fit_catboost(embeddings[fit_rows], y_train)
    r2 = r2_score(y_test, model.predict(embeddings[score_rows]))
    return {
        "r2": float(r2),
        "n_train": n_train,
        "n_test": n_test,
        "n_winsorized": n_clipped,
    }


def run_zeroshot_bootstrap(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = REPORTED_TIERS,
    device: str = "auto",
    models: list[str] | None = None,
) -> dict:
    """Bootstrap zero-shot transfer across borehole folds.

    Args:
        n_iterations: Number of folds. Uses the same fold sequence as
            `run_bootstrap`, so rows are directly comparable.
        test_fraction: Fraction of boreholes held out per fold.
        tiers: Curation tiers to evaluate. Defaults to the tiers the
            paper reports; procedural and duplicate columns are skipped
            because retraining on them wastes hours.
        device: "auto", "cpu", "cuda", or "mps".
        models: Trainer names from `ALL_TRAINERS`. Defaults to the two
            VAE variants.

    Returns:
        dict with keys: config, per_iteration, summary, elapsed_seconds.
    """
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])

    registry = MODEL_REGISTRY
    if models is not None:
        wanted = set(models)
        registry = [(n, fn) for (n, fn) in ALL_TRAINERS if n in wanted]
        if not registry:
            known = ", ".join(n for n, _ in ALL_TRAINERS)
            raise ValueError(f"No trainers match {models}. Known: {known}")

    names = select_targets(targets, tiers)
    prepared = {}
    for name in names:
        spec = TARGET_REGISTRY[name]
        values = np.asarray(targets[name]["values"], dtype=np.float64)
        mask, n_dropped = apply_valid_range(
            values, np.asarray(targets[name]["valid_mask"], dtype=bool), spec
        )
        if n_dropped:
            print(f"  {spec.label}: dropped {n_dropped} row(s) outside "
                  f"{spec.valid_range} as physically impossible")
        prepared[name] = (values, mask, bool(spec.winsorize))
    features_np = features.numpy().astype(np.float64)
    labels_np = labels.numpy()

    print(f"Zero-shot transfer bootstrap: {n_iterations} folds, "
          f"{len(names)} targets, {len(registry)} model(s)")

    per_iteration: list[dict] = []
    start = time.time()

    for fold in borehole_splits(
        borehole_ids, n_iterations=n_iterations,
        test_fraction=test_fraction, seed=SEED,
    ):
        fold_start = time.time()
        print(f"\n{'='*70}")
        print(f"Fold {fold.iteration+1}/{n_iterations}: "
              f"{fold.n_train_samples} train / {fold.n_test_samples} held-out rows "
              f"({len(fold.test_boreholes)} boreholes)")
        print(f"{'='*70}")

        train_features = features[fold.train_mask]
        train_labels = labels[fold.train_mask]

        fold_result = {
            "iteration": fold.iteration,
            "n_train_samples": fold.n_train_samples,
            "n_test_samples": fold.n_test_samples,
            "n_test_boreholes": len(fold.test_boreholes),
            # How far this fold's held-out boreholes sit from the training
            # set, so a poor fold can be attributed rather than excused.
            "lithology_jaccard": lithology_jaccard(fold, labels_np),
            "energy_distance": energy_distance(fold, features_np),
            "models": [],
        }

        for model_name, trainer in registry:
            t0 = time.time()
            print(f"\n  [{model_name}]")
            try:
                # Passing the full matrix as `test_features` embeds every
                # row with a model fitted only on the training boreholes.
                trained = trainer(train_features, train_labels, features, dev)
                z = trained.embeddings.numpy()
            except Exception as e:
                print(f"    TRAIN ERROR: {e}")
                fold_result["models"].append({"model": model_name, "error": str(e)})
                continue

            per_variable = {}
            for name, (values, valid, wins) in prepared.items():
                try:
                    per_variable[name] = evaluate_zeroshot_transfer(
                        z, values, valid, fold.train_mask, fold.test_mask,
                        winsorize=wins,
                    )
                except Exception as e:
                    per_variable[name] = {"r2": float("nan"), "error": str(e)}
                per_variable[name]["tier"] = TARGET_REGISTRY[name].tier

            metrics = _fold_metrics(model_name, per_variable)
            metrics["per_variable"] = per_variable
            fold_result["models"].append(metrics)
            print(f"    headline R2 (median over targets)="
                  f"{metrics['zeroshot_r2_headline']:.4f} "
                  f"(n={metrics['zeroshot_n_headline']}, "
                  f"{metrics['zeroshot_n_headline_negative']} negative)  "
                  f"({time.time()-t0:.1f}s)")

        fold_result["elapsed_seconds"] = time.time() - fold_start
        per_iteration.append(fold_result)

    elapsed = time.time() - start
    summary = _summarize(per_iteration)

    print(f"\n{'='*70}")
    print(f"Zero-shot transfer bootstrap complete in {elapsed/60:.1f} min")
    print(f"{'='*70}")
    _print_summary(summary)

    return {
        "config": {
            "n_iterations": n_iterations,
            "test_fraction": test_fraction,
            "tiers": list(tiers),
            "targets": names,
            "models": [n for n, _ in registry],
            "seed": SEED,
        },
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }


def _fold_metrics(model_name: str, per_variable: dict) -> dict:
    """Summarise one fold's per-target R^2 within each curation tier.

    The fold statistic is the **median** across targets, not the mean.
    Out-of-borehole R^2 is unbounded below: when a held-out borehole
    sits outside the range the regressor was fitted on, a single target
    can score in the hundreds or thousands negative and drag a mean to a
    number that describes that one extrapolation failure rather than the
    representation. A mean over targets is reported alongside so the
    blow-ups stay visible instead of being quietly smoothed away, and
    `n_negative` counts how many targets failed outright.
    """
    by_tier: dict[str, list[float]] = {}
    for res in per_variable.values():
        r2 = res.get("r2")
        if r2 is None or not np.isfinite(r2):
            continue
        by_tier.setdefault(res["tier"], []).append(float(r2))

    metrics = {"model": model_name}
    for label, tiers in (("headline", HEADLINE_TIERS), ("reported", REPORTED_TIERS)):
        vals = [v for t in tiers for v in by_tier.get(t, [])]
        arr = np.array(vals) if vals else np.array([np.nan])
        metrics[f"zeroshot_r2_{label}"] = float(np.median(arr))
        metrics[f"zeroshot_r2_{label}_mean"] = float(np.mean(arr))
        metrics[f"zeroshot_n_{label}"] = len(vals)
        metrics[f"zeroshot_n_{label}_negative"] = int(np.sum(arr < 0)) if vals else 0
    return metrics


def _summarize(per_iteration: list[dict]) -> dict:
    """Median, IQR and 95% CI per model, plus per-target distributions."""
    model_names = {m["model"] for it in per_iteration for m in it["models"]}
    summary: dict[str, dict] = {}

    for name in sorted(model_names):
        entry: dict = {}
        for key in (
            "zeroshot_r2_headline",
            "zeroshot_r2_headline_mean",
            "zeroshot_r2_reported",
            "zeroshot_r2_reported_mean",
        ):
            values = [
                m[key]
                for it in per_iteration
                for m in it["models"]
                if m["model"] == name and np.isfinite(m.get(key, float("nan")))
            ]
            if values:
                entry[key] = _distribution(np.array(values))

        per_target: dict[str, dict] = {}
        for it in per_iteration:
            for m in it["models"]:
                if m["model"] != name:
                    continue
                for target, res in m.get("per_variable", {}).items():
                    if np.isfinite(res.get("r2", float("nan"))):
                        per_target.setdefault(target, []).append(res["r2"])
        entry["per_target"] = {
            t: _distribution(np.array(v)) for t, v in sorted(per_target.items())
        }
        summary[name] = entry

    return summary


def _distribution(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "iqr_low": float(np.percentile(arr, 25)),
        "iqr_high": float(np.percentile(arr, 75)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
    }


def _print_summary(summary: dict) -> None:
    for model, entry in summary.items():
        print(f"\n{model}")
        for key in (
            "zeroshot_r2_headline",
            "zeroshot_r2_headline_mean",
            "zeroshot_r2_reported",
            "zeroshot_r2_reported_mean",
        ):
            d = entry.get(key)
            if d:
                print(f"  {key:<30} median {d['median']:.4f} "
                      f"[IQR {d['iqr_low']:.4f}, {d['iqr_high']:.4f}]  n={d['n']}")
        if entry.get("per_target"):
            print(f"  {'target':<46} {'median':>7} {'IQR':>19}")
            for target, d in entry["per_target"].items():
                label = TARGET_REGISTRY[target].label if target in TARGET_REGISTRY else target
                print(f"    {label[:44]:<44} {d['median']:7.3f} "
                      f"[{d['iqr_low']:7.3f}, {d['iqr_high']:7.3f}]")
