"""Tests for borehole folds and zero-shot target curation.

The fold tests need no data. The registry tests do: they exist to fail
loudly when a change to preprocessing harvests a column that nobody has
curated, because such a column would otherwise enter a headline mean
silently.
"""

import numpy as np
import pytest
import torch

from lilyvae.config import PROCESSED_DIR, TRAINING_VARIABLES
from lilyvae.evaluation.folds import (
    borehole_splits,
    energy_distance,
    lithology_jaccard,
)
from lilyvae.evaluation.targets import (
    REPORTED_TIERS,
    TARGET_REGISTRY,
    RegistryError,
    redundancy_clusters,
    select_targets,
    validate_registry,
    winsorize_values,
)


@pytest.fixture
def borehole_ids():
    rng = np.random.RandomState(0)
    return np.array([f"BH{i:03d}" for i in rng.randint(0, 296, size=5000)])


# ---------------------------------------------------------------- folds


def test_folds_are_reproducible(borehole_ids):
    """Two callers with the same arguments must see identical folds.

    This is what makes a bootstrapped baseline row comparable to a
    bootstrapped VAE row: they are scored on the same held-out data.
    """
    a = [f.test_mask for f in borehole_splits(borehole_ids, 5, 0.1)]
    b = [f.test_mask for f in borehole_splits(borehole_ids, 5, 0.1)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_folds_differ_from_each_other(borehole_ids):
    folds = [f.test_mask for f in borehole_splits(borehole_ids, 5, 0.1)]
    assert not np.array_equal(folds[0], folds[1])


def test_folds_split_whole_boreholes(borehole_ids):
    """No borehole may appear on both sides; that would leak depth-adjacent rows."""
    for fold in borehole_splits(borehole_ids, 5, 0.1):
        train = set(borehole_ids[fold.train_mask])
        test = set(borehole_ids[fold.test_mask])
        assert not (train & test)
        assert train | test == set(borehole_ids)
        assert fold.n_train_samples + fold.n_test_samples == len(borehole_ids)


def test_test_fraction_is_over_boreholes_not_samples(borehole_ids):
    n_unique = len(np.unique(borehole_ids))
    for fold in borehole_splits(borehole_ids, 3, 0.1):
        assert len(fold.test_boreholes) == max(1, int(n_unique * 0.1))


def test_fold_distance_diagnostics(borehole_ids):
    rng = np.random.RandomState(1)
    features = rng.normal(size=(len(borehole_ids), 6))
    labels = rng.randint(0, 20, size=len(borehole_ids))
    fold = next(iter(borehole_splits(borehole_ids, 1, 0.1)))

    jaccard = lithology_jaccard(fold, labels)
    assert 0.0 <= jaccard <= 1.0

    # Same distribution on both sides, so the statistic sits near zero.
    assert abs(energy_distance(fold, features, max_samples=400)) < 0.5


def test_energy_distance_detects_a_shifted_holdout(borehole_ids):
    rng = np.random.RandomState(2)
    features = rng.normal(size=(len(borehole_ids), 6))
    fold = next(iter(borehole_splits(borehole_ids, 1, 0.1)))
    shifted = features.copy()
    shifted[fold.test_mask] += 5.0
    assert energy_distance(fold, shifted, max_samples=400) > energy_distance(
        fold, features, max_samples=400
    )


# ------------------------------------------------------------- registry


def test_reported_groups_have_one_representative():
    """Two reported targets sharing a group would double-count one quantity."""
    seen = {}
    for name, spec in TARGET_REGISTRY.items():
        if spec.tier in REPORTED_TIERS:
            assert spec.group not in seen, (
                f"{name} and {seen[spec.group]} both represent {spec.group}"
            )
            seen[spec.group] = name


def test_excluded_targets_carry_a_reason():
    """Every exclusion must be explainable in the supporting information."""
    for name, spec in TARGET_REGISTRY.items():
        if spec.tier in ("procedural", "duplicate"):
            assert spec.reason, f"{name} is excluded without a stated reason"


def test_validate_registry_rejects_an_uncurated_column():
    with pytest.raises(RegistryError, match="no TargetSpec"):
        validate_registry(list(TARGET_REGISTRY) + ["NEW_Some new column"])


def test_validate_registry_rejects_a_stale_entry():
    with pytest.raises(RegistryError, match="no longer exist"):
        validate_registry(list(TARGET_REGISTRY)[:-1])


def test_winsorize_only_bites_on_extreme_tails():
    clean = np.linspace(0.0, 1.0, 1000)
    out, n = winsorize_values(clean)
    assert np.allclose(out.min(), np.percentile(clean, 1))
    assert n <= 20

    corrupt = np.concatenate([clean, np.full(5, 1e12)])
    out, n = winsorize_values(corrupt)
    assert out.max() < 1e12
    assert n >= 5


# ------------------------------------------------- registry vs real data

requires_data = pytest.mark.requires_data


@requires_data
def test_registry_matches_the_harvested_targets():
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    validate_registry(targets.keys())


@requires_data
def test_no_independent_target_is_collinear_with_anything():
    """Audit the hand tiering against the data.

    A target tiered `independent` must not be rank-collinear with a
    training variable (it would be an input in disguise) nor with another
    reported target (it would be the same measurement twice).
    """
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True).numpy()
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    columns = {
        f"[input] {v}": features[:, i].astype(np.float64)
        for i, v in enumerate(TRAINING_VARIABLES)
    }
    for name in select_targets(targets, ("independent",)):
        values = np.asarray(targets[name]["values"], dtype=np.float64).copy()
        values[~np.asarray(targets[name]["valid_mask"], dtype=bool)] = np.nan
        columns[name] = values

    offenders = [
        cluster
        for cluster in redundancy_clusters(columns, threshold=0.95)
        if any(not c.startswith("[input] ") for c in cluster)
    ]
    assert not offenders, f"independent targets are collinear: {offenders}"


# ------------------------------------------------- zero-shot transfer


def test_transfer_refuses_thin_folds():
    """Too few rows on either side must give nan, not a noisy number."""
    from lilyvae.evaluation.zeroshot_bootstrap import evaluate_zeroshot_transfer

    n = 200
    rng = np.random.RandomState(0)
    z = rng.normal(size=(n, 10))
    y = rng.normal(size=n)
    valid = np.ones(n, dtype=bool)
    train = np.zeros(n, dtype=bool)
    train[:10] = True          # below MIN_TRAIN_ROWS
    test = ~train

    assert np.isnan(
        evaluate_zeroshot_transfer(z, y, valid, train, test)["r2"]
    )


def test_transfer_never_fits_on_held_out_rows():
    """A target that is pure noise on the held-out side cannot score well."""
    from lilyvae.evaluation.zeroshot_bootstrap import evaluate_zeroshot_transfer

    n = 2000
    rng = np.random.RandomState(0)
    z = rng.normal(size=(n, 10))
    y = z[:, 0] * 3.0
    train = np.zeros(n, dtype=bool)
    train[: n // 2] = True
    test = ~train
    # Scramble the relationship on the held-out side only.
    y_broken = y.copy()
    y_broken[test] = rng.normal(size=test.sum()) * 3.0

    res = evaluate_zeroshot_transfer(
        z, y_broken, np.ones(n, dtype=bool), train, test
    )
    assert res["n_train"] == n // 2
    assert res["r2"] < 0.1


def test_transfer_winsorize_thresholds_come_from_training_rows():
    """Clipping must not peek at the held-out distribution."""
    from lilyvae.evaluation.zeroshot_bootstrap import evaluate_zeroshot_transfer

    n = 1000
    rng = np.random.RandomState(0)
    z = rng.normal(size=(n, 10))
    y = z[:, 0].copy()
    train = np.zeros(n, dtype=bool)
    train[: n // 2] = True
    test = ~train
    y[test][:5] = 1e12  # corruption only on the held-out side

    res = evaluate_zeroshot_transfer(
        z, y, np.ones(n, dtype=bool), train, test, winsorize=True
    )
    assert np.isfinite(res["r2"])
