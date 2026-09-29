"""Does the borehole holdout cost anything a random row split does not?

Section 4.4 claims it does not, on the grounds that what separates these
protocols from the in-sample numbers is whether the regressor is scored
on rows it was fitted on, and not how the rows are divided. The numbers
it quotes are not reproducible from anything in this repository, so this
measures the comparison directly.

The encoder is held fixed, at the checkpoint trained on the full
dataset, and only the downstream regressor's split changes. That is
what the claim is about: two protocols for scoring the same
representation. Retraining the encoder in one arm and not the other
would confound the split with what the encoder saw.

``sample_split``    10% of rows held out at random. Rows are 20 cm bins
                    along a core, so a held-out bin usually has its
                    neighbours in training, and those neighbours are
                    near-duplicates of it.
``borehole_split``  10% of boreholes held out, the protocol used
                    everywhere else in the paper.

If the two come out close, Section 4.4 stands. If the sample split
scores much higher, the borehole holdout is measuring something the
random split cannot, and Section 3.3 is right that a sample split leaks.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lilyvae.config import CHECKPOINT_DIR, PROCESSED_DIR, SEED
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
from lilyvae.evaluation.depth_bootstrap import _prepare
from sklearn.metrics import r2_score

ARMS = ("sample_split", "borehole_split")


def _score(
    embeddings: np.ndarray,
    values: np.ndarray,
    fit_rows: np.ndarray,
    score_rows: np.ndarray,
    winsorize: bool = False,
) -> tuple[float, int, int]:
    n_train, n_test = int(fit_rows.sum()), int(score_rows.sum())
    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        return float("nan"), n_train, n_test
    y_train, y_test = values[fit_rows], values[score_rows]
    if winsorize:
        # The whole-core magnetic columns carry records that are orders
        # of magnitude out. Left alone they produce R^2 of -1e19 here,
        # which is a statement about a handful of rows. Thresholds come
        # from the training side only, as everywhere else.
        lo, hi = np.percentile(y_train, [1.0, 99.0])
        y_train, y_test = np.clip(y_train, lo, hi), np.clip(y_test, lo, hi)
    if np.ptp(y_test) == 0:
        return float("nan"), n_train, n_test
    model = _fit_catboost(embeddings[fit_rows], y_train)
    return (
        float(r2_score(y_test, model.predict(embeddings[score_rows]))),
        n_train,
        n_test,
    )


def run_split_comparison(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = HEADLINE_TIERS,
) -> dict:
    """Score every target under both splits, over matched iterations."""
    embeddings = torch.load(
        CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True
    ).numpy()
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids = np.asarray(metadata["borehole_ids"])
    names = select_targets(targets, tiers)
    prepared = _prepare(targets, names)
    rng = np.random.RandomState(SEED)

    print(f"Split comparison: {n_iterations} iterations, {len(names)} targets, "
          f"encoder held fixed")

    per_iteration, start = [], time.time()
    folds = list(borehole_splits(borehole_ids, n_iterations, test_fraction, seed=SEED))
    for fold in folds:
        # The random split holds out the same number of rows the borehole
        # split does, so the two arms differ in which rows, not how many.
        n_rows = len(borehole_ids)
        n_holdout = int(fold.test_mask.sum())
        picked = rng.choice(n_rows, size=n_holdout, replace=False)
        sample_test = np.zeros(n_rows, dtype=bool)
        sample_test[picked] = True

        entry = {"iteration": fold.iteration, "n_holdout_rows": n_holdout,
                 "per_variable": {}}
        for name in names:
            values, valid, winsorize = prepared[name]
            row = {}
            for arm, test_mask in (("sample_split", sample_test),
                                   ("borehole_split", fold.test_mask)):
                r2, n_tr, n_te = _score(
                    embeddings, values, valid & ~test_mask, valid & test_mask,
                    winsorize,
                )
                row[arm] = r2
                row[f"n_train_{arm}"] = n_tr
                row[f"n_test_{arm}"] = n_te
            entry["per_variable"][name] = row
        per_iteration.append(entry)
        done = [entry["per_variable"][n] for n in names]
        print(f"  fold {fold.iteration + 1}/{n_iterations}  "
              f"sample {np.nanmedian([d['sample_split'] for d in done]):+.3f}  "
              f"borehole {np.nanmedian([d['borehole_split'] for d in done]):+.3f}",
              flush=True)

    summary = {}
    for name in names:
        row = {"label": TARGET_REGISTRY[name].label}
        for arm in ARMS:
            vals = [
                it["per_variable"][name][arm]
                for it in per_iteration
                if np.isfinite(it["per_variable"][name][arm])
            ]
            row[arm] = float(np.median(vals)) if vals else float("nan")
            row[f"n_folds_{arm}"] = len(vals)
        summary[name] = row

    return {
        "config": {"n_iterations": n_iterations, "test_fraction": test_fraction,
                   "tiers": list(tiers), "encoder": "unsupervised, fixed"},
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": time.time() - start,
    }
