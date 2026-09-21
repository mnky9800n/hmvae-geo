"""Borehole-level train/test folds shared by every evaluation.

Individual boreholes are geologically non-exchangeable: one borehole can
carry lithologies that appear nowhere else in the database. A random
sample-level split therefore leaks information, and any *single*
borehole-level split is biased by whichever boreholes happen to land in
the held-out set.

The answer to that is not to retreat to in-sample evaluation. It is to
never report a single split: every held-out number in the paper is a
median and IQR over the same N folds, so the spread across folds *is*
the measurement of borehole non-exchangeability rather than a defect
hidden by it.

This module is the single source of those folds. Every evaluator
(`bootstrap`, `zeroshot_bootstrap`, `depth_confound`, `raw_baselines`)
consumes `borehole_splits` with the same arguments, so rows in the
paper's tables are computed on identical data and are directly
comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from lilyvae.config import SEED


@dataclass(frozen=True)
class Fold:
    """One borehole-level train/test split.

    Attributes:
        iteration: Zero-based fold index.
        train_mask: Boolean mask over samples, True for training rows.
        test_mask: Boolean mask over samples, True for held-out rows.
        train_boreholes: Sorted borehole IDs in the training set.
        test_boreholes: Sorted borehole IDs held out.
    """

    iteration: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_boreholes: np.ndarray
    test_boreholes: np.ndarray

    @property
    def n_train_samples(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test_samples(self) -> int:
        return int(self.test_mask.sum())


def borehole_splits(
    borehole_ids: np.ndarray,
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    seed: int = SEED,
) -> Iterator[Fold]:
    """Yield reproducible borehole-level train/test folds.

    The random state is constructed once and permuted once per iteration,
    so folds differ from each other while the whole sequence is fully
    determined by ``seed``. Calling this twice with identical arguments
    yields identical folds, which is what makes results from different
    evaluators comparable row by row.

    Args:
        borehole_ids: Per-sample borehole identifier, length n_samples.
            Row order must match the feature/label/target tensors.
        n_iterations: Number of folds to yield.
        test_fraction: Fraction of *boreholes* (not samples) held out.
        seed: Seed for the permutation sequence.

    Yields:
        Fold instances in iteration order.
    """
    borehole_ids = np.asarray(borehole_ids)
    unique_boreholes = np.unique(borehole_ids)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))
    rng = np.random.RandomState(seed)

    for iteration in range(n_iterations):
        perm = rng.permutation(unique_boreholes)
        test_boreholes = perm[:n_test]
        test_mask = np.isin(borehole_ids, test_boreholes)
        train_mask = ~test_mask
        yield Fold(
            iteration=iteration,
            train_mask=train_mask,
            test_mask=test_mask,
            train_boreholes=np.sort(perm[n_test:]),
            test_boreholes=np.sort(test_boreholes),
        )


def lithology_jaccard(fold: Fold, labels: np.ndarray) -> float:
    """Jaccard overlap between the lithology sets of train and held-out rows.

    1.0 means the held-out boreholes contain no lithology absent from
    training; low values mean the fold asks the model to generalise to
    rock types it never saw.
    """
    labels = np.asarray(labels)
    train_classes = set(np.unique(labels[fold.train_mask]).tolist())
    test_classes = set(np.unique(labels[fold.test_mask]).tolist())
    union = train_classes | test_classes
    if not union:
        return float("nan")
    return len(train_classes & test_classes) / len(union)


def energy_distance(
    fold: Fold,
    features: np.ndarray,
    max_samples: int = 2000,
    seed: int = SEED,
) -> float:
    """Energy distance between train and held-out rows in feature space.

    The standard two-sample statistic
    ``2 E|X-Y| - E|X-X'| - E|Y-Y'``, which is zero exactly when the two
    samples come from the same distribution. Both sides are subsampled
    to ``max_samples`` rows because the pairwise terms are O(n^2); the
    subsample is seeded so the value is reproducible.

    Used to ask whether a fold's held-out R^2 is explained by how far
    that fold's boreholes sit from the training distribution.
    """
    features = np.asarray(features, dtype=np.float64)
    rng = np.random.RandomState(seed + fold.iteration)

    def _subsample(mask: np.ndarray) -> np.ndarray:
        rows = features[mask]
        if len(rows) > max_samples:
            rows = rows[rng.choice(len(rows), max_samples, replace=False)]
        return rows

    x = _subsample(fold.train_mask)
    y = _subsample(fold.test_mask)
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
        # Chunked to keep the pairwise matrix bounded in memory.
        total, count = 0.0, 0
        for start in range(0, len(a), 512):
            block = a[start : start + 512]
            d = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=-1)
            total += float(d.sum())
            count += d.size
        return total / count

    return 2.0 * _mean_dist(x, y) - _mean_dist(x, x) - _mean_dist(y, y)


@dataclass(frozen=True)
class NestedSplit:
    """A borehole pool reserved for model selection, and one held back.

    Searching over configurations and keeping the best is itself a
    fitting procedure. If the search reads the same folds the paper
    reports, the reported number is a maximum over many draws rather
    than an estimate, and it will not survive contact with new data.
    That is the adaptive-overfitting version of the in-sample problem,
    and the only reliable defence is boreholes the search never sees.

    `search_boreholes` is what a tuning loop may use, through
    `borehole_splits` restricted to it. `report_boreholes` is scored once,
    at the end, by whichever configuration the search settled on.
    """

    search_mask: np.ndarray
    report_mask: np.ndarray
    search_boreholes: np.ndarray
    report_boreholes: np.ndarray


def nested_split(
    borehole_ids: np.ndarray,
    report_fraction: float = 0.2,
    seed: int = SEED,
) -> NestedSplit:
    """Reserve a fraction of boreholes that model selection may not touch.

    Args:
        borehole_ids: Per-sample borehole identifier.
        report_fraction: Fraction of boreholes held back for the final
            report.
        seed: Seed for the partition. Deliberately offset from the fold
            seed so the reporting pool is not simply the first fold of
            the ordinary bootstrap.
    """
    borehole_ids = np.asarray(borehole_ids)
    unique = np.unique(borehole_ids)
    n_report = max(1, int(len(unique) * report_fraction))
    perm = np.random.RandomState(seed + 9973).permutation(unique)
    report_boreholes = np.sort(perm[:n_report])
    search_boreholes = np.sort(perm[n_report:])
    report_mask = np.isin(borehole_ids, report_boreholes)
    return NestedSplit(
        search_mask=~report_mask,
        report_mask=report_mask,
        search_boreholes=search_boreholes,
        report_boreholes=report_boreholes,
    )
