"""How much of the result is the representation, and how much is the task?

A reviewer's central objection was that the difficulty of these tasks is
never established: the paper shows good lithology classification and
good prediction of held-out variables from a ten-dimensional embedding,
without showing what the same predictors achieve on the six raw inputs.
Without that comparison a strong number could mean a strong
representation or an easy task, and there is no way to tell which.

The comparison is also the honest one to make. The encoder maps six
standardised inputs to a ten-dimensional latent, so it cannot add
information that the six inputs do not already carry. If the raw inputs
do as well, the claim for the embedding cannot be predictive accuracy,
and has to rest on what the representation makes explicit: a compact,
continuous, label-free space in which lithology separates and in which
directions correspond to interpretable geology.

Every arm here is scored on held-out boreholes over the same folds as
the rest of the evaluation, so the comparison is between feature sets
rather than between protocols.

Arms
----
``raw6``       the six standardised training variables
``pca``        principal components of those six
``embedding``  the VAE latent, retrained per fold
``knn``        a k-nearest-neighbour predictor on the raw six, which sets
               the scale for how much of the signal is plain local
               similarity rather than a learned structure
"""

from __future__ import annotations

import time

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC

from lilyvae.config import LATENT_DIM, PROCESSED_DIR, SEED
from lilyvae.evaluation.bootstrap import ALL_TRAINERS
from lilyvae.evaluation.folds import borehole_splits, energy_distance, lithology_jaccard
from lilyvae.evaluation.targets import REPORTED_TIERS, TARGET_REGISTRY, select_targets
from lilyvae.evaluation.zeroshot_bootstrap import (
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    _fold_metrics,
    evaluate_zeroshot_transfer,
)
from lilyvae.training.trainer import get_device

#: Neighbours for the kNN arm, matching the earlier rebuttal script.
KNN_K = 10

#: Feature sets compared. "embedding" is resolved per fold by training a
#: VAE; the others are closed-form functions of the raw inputs.
DEFAULT_ARMS = ("raw6", "pca", "embedding")


def _build_arm(
    arm: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: np.ndarray,
    device,
) -> np.ndarray:
    """Return the per-sample feature matrix for one arm.

    Anything fitted (PCA, the VAE) is fitted on training boreholes only
    and then applied to every row, so held-out rows never influence the
    representation they are scored in.
    """
    raw = features.numpy().astype(np.float64)

    if arm == "raw6":
        return raw

    if arm == "pca":
        # Six inputs admit at most six components, so this is a rotation
        # rather than a compression; it is included because the earlier
        # draft reported PCA as a baseline.
        pca = PCA(n_components=min(LATENT_DIM, raw.shape[1]), random_state=SEED)
        pca.fit(raw[train_mask])
        return pca.transform(raw)

    if arm == "embedding":
        trainer = dict(ALL_TRAINERS)["unsupervised_vae"]
        trained = trainer(
            features[train_mask], labels[train_mask], features, device
        )
        return trained.embeddings.numpy().astype(np.float64)

    raise ValueError(f"unknown arm {arm!r}")


def evaluate_classification_transfer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    top_n: int = 10,
) -> dict:
    """One-versus-rest linear SVM AUC, fitted on train rows, scored on test.

    The in-sample evaluator in `evaluation/classification.py` fits and
    scores the same rows. This is its held-out counterpart. Classes with
    fewer than two examples on either side cannot be scored and are
    skipped; the count of scored classes is returned so a macro average
    over a shrinking class set is never mistaken for an improvement.
    """
    per_class = {}
    counts = np.bincount(y_train, minlength=int(max(y_train.max(), y_test.max())) + 1)

    for cls in np.unique(y_train):
        train_bin = (y_train == cls).astype(int)
        test_bin = (y_test == cls).astype(int)
        if train_bin.sum() < 2 or (1 - train_bin).sum() < 2:
            continue
        if test_bin.sum() < 2 or (1 - test_bin).sum() < 2:
            continue
        try:
            svm = LinearSVC(max_iter=10000, random_state=SEED, dual="auto")
            svm.fit(x_train, train_bin)
            per_class[int(cls)] = float(
                roc_auc_score(test_bin, svm.decision_function(x_test))
            )
        except Exception:
            continue

    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    top = np.argsort(counts)[::-1][:top_n]
    return {
        "auc_macro": macro,
        "n_classes_evaluated": len(per_class),
        "top_class_auc": {int(c): per_class[int(c)] for c in top if int(c) in per_class},
        "per_class_auc": per_class,
    }


def evaluate_supervised_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    top_n: int = 10,
) -> dict:
    """A directly supervised classifier, the upper bound on this task.

    This is the baseline the reviewer asked for: a model that sees the
    labels and the raw inputs together, with no representation learning
    in between. It bounds what any label-free pipeline could achieve.
    """
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1, verbose=0,
        random_seed=SEED, loss_function="MultiClass",
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)
    model_classes = clf.classes_.astype(int)

    per_class = {}
    counts = np.bincount(y_train, minlength=int(max(y_train.max(), y_test.max())) + 1)
    for col, cls in enumerate(model_classes):
        test_bin = (y_test == cls).astype(int)
        if test_bin.sum() < 2 or (1 - test_bin).sum() < 2:
            continue
        try:
            per_class[int(cls)] = float(roc_auc_score(test_bin, proba[:, col]))
        except Exception:
            continue

    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    top = np.argsort(counts)[::-1][:top_n]
    return {
        "auc_macro": macro,
        "n_classes_evaluated": len(per_class),
        "top_class_auc": {int(c): per_class[int(c)] for c in top if int(c) in per_class},
    }


def evaluate_knn_transfer(
    raw: np.ndarray,
    target_values: np.ndarray,
    valid_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    k: int = KNN_K,
) -> dict:
    """k-nearest-neighbour regression on the raw inputs, held out by borehole.

    Sets the scale for how much of the zero-shot signal is available from
    plain local similarity in the six measurements.
    """
    fit_rows = valid_mask & train_mask
    score_rows = valid_mask & test_mask
    n_train, n_test = int(fit_rows.sum()), int(score_rows.sum())
    if n_train < max(MIN_TRAIN_ROWS, k) or n_test < MIN_TEST_ROWS:
        return {"r2": float("nan"), "n_train": n_train, "n_test": n_test}

    y_train = np.asarray(target_values, dtype=np.float64)[fit_rows]
    y_test = np.asarray(target_values, dtype=np.float64)[score_rows]
    if np.ptp(y_test) == 0:
        return {"r2": float("nan"), "n_train": n_train, "n_test": n_test}

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(raw[fit_rows], y_train)
    return {
        "r2": float(r2_score(y_test, model.predict(raw[score_rows]))),
        "n_train": n_train,
        "n_test": n_test,
    }


def run_raw_baselines(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = REPORTED_TIERS,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    device: str = "auto",
    include_knn: bool = True,
    include_supervised: bool = True,
) -> dict:
    """Compare feature sets on held-out boreholes across the shared folds."""
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])

    names = select_targets(targets, tiers)
    prepared = {
        name: (
            np.asarray(targets[name]["values"], dtype=np.float64),
            np.asarray(targets[name]["valid_mask"], dtype=bool),
            bool(TARGET_REGISTRY[name].winsorize),
        )
        for name in names
    }
    raw = features.numpy().astype(np.float64)
    labels_np = labels.numpy()

    print(f"Raw-input baselines: {n_iterations} folds, {len(names)} targets, "
          f"arms={', '.join(arms)}")

    per_iteration: list[dict] = []
    start = time.time()

    for fold in borehole_splits(
        borehole_ids, n_iterations=n_iterations,
        test_fraction=test_fraction, seed=SEED,
    ):
        fold_start = time.time()
        print(f"\n{'='*70}")
        print(f"Fold {fold.iteration+1}/{n_iterations}")
        print(f"{'='*70}")

        y_train = labels_np[fold.train_mask]
        y_test = labels_np[fold.test_mask]

        fold_result = {
            "iteration": fold.iteration,
            "n_train_samples": fold.n_train_samples,
            "n_test_samples": fold.n_test_samples,
            "lithology_jaccard": lithology_jaccard(fold, labels_np),
            "energy_distance": energy_distance(fold, raw),
            "models": [],
        }

        for arm in arms:
            t0 = time.time()
            print(f"\n  [{arm}]")
            try:
                x = _build_arm(arm, features, labels, fold.train_mask, dev)
            except Exception as e:
                print(f"    ARM ERROR: {e}")
                fold_result["models"].append({"model": arm, "error": str(e)})
                continue

            per_variable = {}
            for name, (values, valid, wins) in prepared.items():
                try:
                    per_variable[name] = evaluate_zeroshot_transfer(
                        x, values, valid, fold.train_mask, fold.test_mask,
                        winsorize=wins,
                    )
                except Exception as e:
                    per_variable[name] = {"r2": float("nan"), "error": str(e)}
                per_variable[name]["tier"] = TARGET_REGISTRY[name].tier

            metrics = _fold_metrics(arm, per_variable)
            metrics["per_variable"] = per_variable
            try:
                cls = evaluate_classification_transfer(
                    x[fold.train_mask], y_train, x[fold.test_mask], y_test
                )
                metrics["auc_macro"] = cls["auc_macro"]
                metrics["n_classes_evaluated"] = cls["n_classes_evaluated"]
            except Exception as e:
                print(f"    classification ERROR: {e}")

            fold_result["models"].append(metrics)
            print(f"    zero-shot median R2={metrics['zeroshot_r2_headline']:.4f}  "
                  f"AUC={metrics.get('auc_macro', float('nan')):.4f}  "
                  f"({time.time()-t0:.1f}s)")

        if include_knn:
            t0 = time.time()
            print("\n  [knn]")
            per_variable = {}
            for name, (values, valid, _) in prepared.items():
                try:
                    per_variable[name] = evaluate_knn_transfer(
                        raw, values, valid, fold.train_mask, fold.test_mask
                    )
                except Exception as e:
                    per_variable[name] = {"r2": float("nan"), "error": str(e)}
                per_variable[name]["tier"] = TARGET_REGISTRY[name].tier
            metrics = _fold_metrics("knn", per_variable)
            metrics["per_variable"] = per_variable
            fold_result["models"].append(metrics)
            print(f"    zero-shot median R2={metrics['zeroshot_r2_headline']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

        if include_supervised:
            t0 = time.time()
            print("\n  [supervised_raw6]")
            try:
                sup = evaluate_supervised_classifier(
                    raw[fold.train_mask], y_train, raw[fold.test_mask], y_test
                )
                fold_result["models"].append({
                    "model": "supervised_raw6",
                    "auc_macro": sup["auc_macro"],
                    "n_classes_evaluated": sup["n_classes_evaluated"],
                })
                print(f"    AUC={sup['auc_macro']:.4f}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    ERROR: {e}")
                fold_result["models"].append(
                    {"model": "supervised_raw6", "error": str(e)}
                )

        fold_result["elapsed_seconds"] = time.time() - fold_start
        per_iteration.append(fold_result)

    elapsed = time.time() - start
    summary = _summarize(per_iteration)
    _print_summary(summary)

    return {
        "config": {
            "n_iterations": n_iterations,
            "test_fraction": test_fraction,
            "tiers": list(tiers),
            "arms": list(arms),
            "include_knn": include_knn,
            "include_supervised": include_supervised,
            "targets": names,
            "seed": SEED,
        },
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }


def _summarize(per_iteration: list[dict]) -> dict:
    keys = ("zeroshot_r2_headline", "zeroshot_r2_reported", "auc_macro")
    models = {m["model"] for it in per_iteration for m in it["models"]}
    summary: dict[str, dict] = {}
    for name in sorted(models):
        entry: dict = {}
        for key in keys:
            vals = [
                m[key] for it in per_iteration for m in it["models"]
                if m["model"] == name and np.isfinite(m.get(key, float("nan")))
            ]
            if vals:
                arr = np.array(vals)
                entry[key] = {
                    "median": float(np.median(arr)),
                    "mean": float(np.mean(arr)),
                    "iqr_low": float(np.percentile(arr, 25)),
                    "iqr_high": float(np.percentile(arr, 75)),
                    "n": int(arr.size),
                }
        summary[name] = entry
    return summary


def _print_summary(summary: dict) -> None:
    print(f"\n{'='*70}")
    print(f"{'arm':<20} {'zero-shot R2 (median)':>24} {'AUC (median)':>16}")
    print(f"{'='*70}")
    for name, entry in summary.items():
        zs = entry.get("zeroshot_r2_headline", {})
        auc = entry.get("auc_macro", {})
        zs_s = (f"{zs['median']:.4f} [{zs['iqr_low']:.3f}, {zs['iqr_high']:.3f}]"
                if zs else "--")
        auc_s = f"{auc['median']:.4f}" if auc else "--"
        print(f"{name:<20} {zs_s:>24} {auc_s:>16}")
