"""Can the lithology label predict what the embedding predicts?

The Discussion says categorisation is a lower-fidelity record of process
than the instruments on the same core. That is the paper's broadest
claim and nothing measured it. This does.

Four sets of inputs, all given to the same regressor and scored on the
same held-out boreholes:

``label``            the lithology label alone, one-hot over the classes
                     present in training. The expert description, and
                     nothing else.
``embedding``        the ten-dimensional latent, as reported elsewhere.
``label_embedding``  both together, which bounds what they carry jointly
                     and says whether the label adds anything the
                     embedding lacks.
``raw6``             the six measurements, as the ceiling the label is
                     a compression of.

A label is a partition of the same core the instruments logged, so the
comparison is not whether geologists are right. It is how much of a
continuously varying property survives being written down as one of 139
names.

Classes absent from a fold's training rows are encoded as all-zero
rather than dropped, because a held-out borehole carrying a lithology
training never saw is the normal case here, not an edge case.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lilyvae.config import CHECKPOINT_DIR, PROCESSED_DIR, SEED
from lilyvae.evaluation.depth_bootstrap import _prepare
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.evaluation.targets import (
    HEADLINE_TIERS,
    TARGET_REGISTRY,
    select_targets,
)
from lilyvae.evaluation.zeroshot_bootstrap import (
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    _fit_catboost,
)
from sklearn.metrics import r2_score

ARMS = ("label", "embedding", "label_embedding", "raw6")


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot over the given classes; unseen classes become all-zero."""
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, lab in enumerate(labels):
        col = index.get(lab)
        if col is not None:
            out[row, col] = 1.0
    return out


def _score(x, values, fit_rows, score_rows, winsorize) -> float:
    n_train, n_test = int(fit_rows.sum()), int(score_rows.sum())
    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        return float("nan")
    y_train, y_test = values[fit_rows], values[score_rows]
    if winsorize:
        lo, hi = np.percentile(y_train, [1.0, 99.0])
        y_train, y_test = np.clip(y_train, lo, hi), np.clip(y_test, lo, hi)
    if np.ptp(y_test) == 0:
        return float("nan")
    model = _fit_catboost(x[fit_rows], y_train)
    return float(r2_score(y_test, model.predict(x[score_rows])))


def run_label_vs_embedding(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = HEADLINE_TIERS,
) -> dict:
    """Score the label, the embedding, both, and the raw inputs."""
    features = torch.load(
        PROCESSED_DIR / "train_features.pt", weights_only=True
    ).numpy().astype(np.float64)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()
    embeddings = torch.load(
        CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True
    ).numpy().astype(np.float64)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids = np.asarray(metadata["borehole_ids"])
    names = select_targets(targets, tiers)
    prepared = _prepare(targets, names)

    print(f"Label against embedding: {n_iterations} folds, {len(names)} targets, "
          f"{len(np.unique(labels))} lithology classes")

    per_iteration, start = [], time.time()
    for fold in borehole_splits(borehole_ids, n_iterations, test_fraction, seed=SEED):
        classes = np.unique(labels[fold.train_mask])
        onehot = _one_hot(labels, classes)
        arms = {
            "label": onehot,
            "embedding": embeddings,
            "label_embedding": np.column_stack([onehot, embeddings]),
            "raw6": features,
        }
        entry = {"iteration": fold.iteration, "n_classes": int(len(classes)),
                 "per_variable": {}}
        for name in names:
            values, valid, winsorize = prepared[name]
            entry["per_variable"][name] = {
                arm: _score(x, values, valid & fold.train_mask,
                            valid & fold.test_mask, winsorize)
                for arm, x in arms.items()
            }
        per_iteration.append(entry)
        meds = {
            arm: np.nanmedian([
                entry["per_variable"][n][arm] for n in names
            ]) for arm in ARMS
        }
        print(f"  fold {fold.iteration + 1}/{n_iterations}  " + "  ".join(
            f"{a} {meds[a]:+.3f}" for a in ARMS), flush=True)

    summary = {}
    for name in names:
        row = {"label_name": TARGET_REGISTRY[name].label}
        for arm in ARMS:
            vals = [
                it["per_variable"][name][arm] for it in per_iteration
                if np.isfinite(it["per_variable"][name][arm])
            ]
            row[arm] = float(np.median(vals)) if vals else float("nan")
        summary[name] = row

    overall = {
        arm: float(np.nanmedian([r[arm] for r in summary.values()]))
        for arm in ARMS
    }
    return {
        "config": {"n_iterations": n_iterations, "test_fraction": test_fraction,
                   "tiers": list(tiers), "seed": SEED},
        "per_iteration": per_iteration,
        "summary": summary,
        "overall": overall,
        "elapsed_seconds": time.time() - start,
    }
