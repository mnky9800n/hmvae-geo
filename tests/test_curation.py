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


# ------------------------------------------------------ raw baselines


def test_classification_transfer_skips_unscorable_classes():
    """A class absent from the held-out side cannot be scored."""
    from lilyvae.evaluation.raw_baselines import evaluate_classification_transfer

    rng = np.random.RandomState(0)
    x_train = rng.normal(size=(400, 6))
    y_train = rng.randint(0, 4, size=400)
    x_test = rng.normal(size=(200, 6))
    y_test = rng.randint(0, 2, size=200)  # classes 2 and 3 never appear

    res = evaluate_classification_transfer(x_train, y_train, x_test, y_test)
    assert res["n_classes_evaluated"] <= 2
    assert set(res["per_class_auc"]) <= {0, 1}


def test_arms_are_fitted_on_training_boreholes_only():
    """PCA must not see held-out rows when it is fitted."""
    import torch as _torch
    from lilyvae.evaluation.raw_baselines import _build_arm

    rng = np.random.RandomState(0)
    raw = rng.normal(size=(600, 6))
    # Give the held-out block a wildly different scale. If PCA were fitted
    # on everything, its components would rotate towards that block.
    train_mask = np.zeros(600, dtype=bool)
    train_mask[:500] = True
    raw[~train_mask] *= 50.0

    features = _torch.tensor(raw, dtype=_torch.float32)
    labels = _torch.tensor(rng.randint(0, 3, size=600))

    projected = _build_arm("pca", features, labels, train_mask, device="cpu")
    # Components fitted on the training block only, so the held-out block
    # still projects to a much larger radius.
    train_radius = np.linalg.norm(projected[train_mask], axis=1).mean()
    test_radius = np.linalg.norm(projected[~train_mask], axis=1).mean()
    assert test_radius > 10 * train_radius


def test_knn_transfer_refuses_thin_folds():
    from lilyvae.evaluation.raw_baselines import evaluate_knn_transfer

    n = 300
    rng = np.random.RandomState(0)
    raw = rng.normal(size=(n, 6))
    y = rng.normal(size=n)
    valid = np.ones(n, dtype=bool)
    train = np.zeros(n, dtype=bool)
    train[:5] = True
    assert np.isnan(
        evaluate_knn_transfer(raw, y, valid, train, ~train)["r2"]
    )


# -------------------------------------------------------- calibration


def test_calibration_recognises_well_calibrated_variances():
    """Residuals drawn at the predicted scale must give coverage near nominal."""
    from lilyvae.evaluation.calibration import evaluate_calibration

    rng = np.random.RandomState(0)
    n, d = 20000, 6
    mu = rng.normal(size=(n, d))
    logvar = rng.uniform(-2.0, 2.0, size=(n, d))
    x = mu + rng.normal(size=(n, d)) * np.exp(0.5 * logvar)

    res = evaluate_calibration(x, mu, logvar, clamp=(-6.0, 6.0))
    assert abs(res["mean_standardised_residual_sq"] - 1.0) < 0.05
    for level in ("0.68", "0.95"):
        cov = res["coverage"][level]
        assert abs(cov["excess"]) < 0.02
    # Learned per-sample variances should beat one variance per column.
    assert res["nll_gain_from_learned_variance"] > 0
    # And larger predicted variance should track larger error.
    assert res["spearman_variance_vs_abs_error"] > 0.3


def test_calibration_detects_overwide_intervals():
    """Variances far too large must show as coverage above nominal."""
    from lilyvae.evaluation.calibration import evaluate_calibration

    rng = np.random.RandomState(1)
    n, d = 20000, 6
    mu = np.zeros((n, d))
    x = rng.normal(size=(n, d)) * 0.01
    logvar = np.zeros((n, d))  # variance 1 against residuals of size 0.01

    res = evaluate_calibration(x, mu, logvar, clamp=(-6.0, 6.0))
    assert res["mean_standardised_residual_sq"] < 0.01
    assert res["coverage"]["0.68"]["empirical"] > 0.99


def test_calibration_measures_clamp_saturation():
    from lilyvae.evaluation.calibration import evaluate_calibration

    n, d = 1000, 6
    mu = np.zeros((n, d))
    x = np.zeros((n, d))
    logvar = np.full((n, d), -6.0)   # every prediction pinned at the floor
    res = evaluate_calibration(x, mu, logvar, clamp=(-6.0, 6.0))
    assert res["fraction_at_clamp_floor"] == 1.0
    assert res["fraction_at_clamp_ceiling"] == 0.0
    # Fully saturated, so there is no ordering to correlate against error.
    assert np.isnan(res["spearman_variance_vs_abs_error"])


def test_decoder_clamp_is_configurable_and_defaults_unchanged():
    from lilyvae.config import LOGVAR_CLAMP_MAX, LOGVAR_CLAMP_MIN
    from lilyvae.model.vae import LilyVAE

    assert LilyVAE().decoder.logvar_clamp == (LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)
    assert LilyVAE(logvar_clamp=(-12.0, 6.0)).decoder.logvar_clamp == (-12.0, 6.0)


def test_calibration_from_model_is_deterministic():
    """Decoding must not reparameterise, or the numbers move between runs."""
    import torch as _torch
    from lilyvae.evaluation.calibration import calibration_from_model
    from lilyvae.model.vae import LilyVAE

    _torch.manual_seed(0)
    vae = LilyVAE()
    x = _torch.randn(512, 6)
    a = calibration_from_model(vae, x)
    b = calibration_from_model(vae, x)
    assert a["nll_heteroscedastic"] == b["nll_heteroscedastic"]
    assert a["mean_standardised_residual_sq"] == b["mean_standardised_residual_sq"]


# --------------------------------------------- physical validity bounds


def test_valid_range_removes_impossible_values():
    """The mechanism that inflated the reported zero-shot scores.

    A single value nine orders of magnitude above the real ones dominates
    the total sum of squares, so memorising it reads as near-perfect R2.
    """
    from lilyvae.evaluation.targets import TargetSpec, apply_valid_range

    spec = TargetSpec("independent", "g", "TCON", "x", valid_range=(0.2, 10.0))
    values = np.array([1.1, 1.3, 0.9, 3.75e9, -0.016])
    valid = np.ones(5, dtype=bool)

    narrowed, dropped = apply_valid_range(values, valid, spec)
    assert dropped == 2
    assert narrowed.tolist() == [True, True, True, False, False]


def test_valid_range_is_a_no_op_without_bounds():
    from lilyvae.evaluation.targets import TargetSpec, apply_valid_range

    spec = TargetSpec("independent", "g", "X", "x")
    valid = np.array([True, True, False])
    narrowed, dropped = apply_valid_range(np.array([1.0, 1e9, 3.0]), valid, spec)
    assert dropped == 0
    assert narrowed is valid


def test_every_bounded_target_has_a_physically_sane_range():
    """Guard against a typo turning a bound into a silent data filter."""
    for name, spec in TARGET_REGISTRY.items():
        if spec.valid_range is None:
            continue
        lo, hi = spec.valid_range
        assert lo < hi, f"{name} has an inverted range"
        assert np.isfinite(lo) and np.isfinite(hi), f"{name} has a non-finite bound"


@requires_data
def test_bounds_discard_only_a_tiny_fraction_of_real_data():
    """Bounds must remove impossible values, not trim the distribution.

    If a bound ever starts discarding a large share of a column it is
    more likely to be wrong than the data is.
    """
    from lilyvae.evaluation.targets import apply_valid_range

    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    for name in select_targets(targets, REPORTED_TIERS):
        spec = TARGET_REGISTRY[name]
        if spec.valid_range is None:
            continue
        values = np.asarray(targets[name]["values"], dtype=np.float64)
        valid = np.asarray(targets[name]["valid_mask"], dtype=bool)
        _, dropped = apply_valid_range(values, valid, spec)
        fraction = dropped / max(int(valid.sum()), 1)
        assert fraction < 0.05, (
            f"{spec.label} loses {fraction:.1%} to its bound {spec.valid_range}; "
            "check the bound before blaming the data"
        )


# ------------------------------------------- training-variable bounds


def test_training_variable_bounds_are_sane():
    from lilyvae.data.preprocessing import TRAINING_VARIABLE_BOUNDS

    for name, (lo, hi) in TRAINING_VARIABLE_BOUNDS.items():
        assert lo < hi, f"{name} has an inverted range"
        assert np.isfinite(lo) and np.isfinite(hi)


def test_susceptibility_and_ngr_are_not_bounded_below_at_zero():
    """Both are legitimately negative and must not be filtered.

    Magnetic susceptibility is negative for diamagnetic carbonate and
    quartz-rich sediment; NGR can go slightly negative after background
    subtraction. Bounding either at zero would delete real measurements.
    """
    from lilyvae.data.preprocessing import TRAINING_VARIABLE_BOUNDS

    assert "magnetic_susceptibility" not in TRAINING_VARIABLE_BOUNDS
    assert "NGR" not in TRAINING_VARIABLE_BOUNDS


@requires_data
def test_training_bounds_would_remove_only_impossible_rows():
    """Applied to the current tensors, the bounds must be a light touch."""
    import pickle

    from lilyvae.data.preprocessing import TRAINING_VARIABLE_BOUNDS

    scaler = pickle.load(open(PROCESSED_DIR / "scaler.pkl", "rb"))
    z = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True).numpy()
    raw = scaler.inverse_transform(z)

    for i, name in enumerate(TRAINING_VARIABLES):
        if name not in TRAINING_VARIABLE_BOUNDS:
            continue
        lo, hi = TRAINING_VARIABLE_BOUNDS[name]
        outside = (raw[:, i] < lo) | (raw[:, i] > hi)
        assert outside.mean() < 0.02, (
            f"{name} loses {outside.mean():.2%} to bound [{lo}, {hi}]; "
            "check the bound before blaming the data"
        )


# ------------------------------------------------------------- search


def test_nested_split_is_disjoint_and_stable():
    from lilyvae.evaluation.folds import nested_split

    rng = np.random.RandomState(0)
    bh = np.array([f"BH{i:03d}" for i in rng.randint(0, 296, size=5000)])

    a = nested_split(bh, report_fraction=0.2)
    b = nested_split(bh, report_fraction=0.2)
    assert np.array_equal(a.report_mask, b.report_mask)
    assert not set(a.search_boreholes) & set(a.report_boreholes)
    assert not (a.search_mask & a.report_mask).any()
    assert (a.search_mask | a.report_mask).all()


def test_search_folds_never_touch_reporting_boreholes():
    """The whole point of the nested split."""
    from lilyvae.evaluation.folds import nested_split

    rng = np.random.RandomState(0)
    bh = np.array([f"BH{i:03d}" for i in rng.randint(0, 296, size=5000)])
    split = nested_split(bh, report_fraction=0.2)
    pool = bh[split.search_mask]

    reserved = set(split.report_boreholes)
    for fold in borehole_splits(pool, 5, 0.1):
        assert not set(pool[fold.test_mask]) & reserved
        assert not set(pool[fold.train_mask]) & reserved


def test_search_config_roundtrips_and_rejects_unknown_keys():
    import pytest as _pytest
    from lilyvae.search.space import SearchConfig, random_config

    c = random_config(np.random.RandomState(0))
    assert SearchConfig.from_dict(c.to_dict()) == c
    with _pytest.raises(ValueError, match="unknown config keys"):
        SearchConfig.from_dict({"latent_dim": 8, "nonsense": 1})


def test_ledger_roundtrips_and_deduplicates(tmp_path):
    from lilyvae.search.ledger import Ledger, Trial
    from lilyvae.search.space import SearchConfig

    ledger = Ledger(tmp_path / "l.jsonl")
    assert ledger.all() == []
    assert ledger.best() is None

    cfg = SearchConfig(latent_dim=16)
    ledger.append(Trial(index=0, config=cfg.to_dict(),
                        ari_median=0.30, auc_median=0.95))
    ledger.append(Trial(index=1, config=SearchConfig(latent_dim=8).to_dict(),
                        ari_median=0.20, auc_median=0.94))

    assert len(ledger.all()) == 2
    assert ledger.next_index() == 2
    assert ledger.best().index == 0
    assert cfg.key() in ledger.seen()


def test_failed_trials_never_win_and_are_excluded_from_seen(tmp_path):
    from lilyvae.search.ledger import Ledger, Trial
    from lilyvae.search.space import SearchConfig

    ledger = Ledger(tmp_path / "l.jsonl")
    bad = SearchConfig(latent_dim=32)
    ledger.append(Trial(index=0, config=bad.to_dict(),
                        ari_median=float("nan"), auc_median=float("nan"),
                        error="boom"))
    assert ledger.best() is None
    # A failure must not block a retry of that configuration.
    assert bad.key() not in ledger.seen()

    ledger.append(Trial(index=1, config=SearchConfig().to_dict(),
                        ari_median=0.25, auc_median=0.95))
    assert ledger.best().index == 1


def test_gmm_component_count_is_not_searchable():
    """ARI is measured with a Gaussian mixture, so its component count
    would move the metric rather than the representation."""
    from lilyvae.search.space import SEARCH_SPACE, SearchConfig

    assert "n_gmm_components" not in SEARCH_SPACE
    assert "n_gmm_components" not in SearchConfig.__dataclass_fields__
    assert "alpha" not in SEARCH_SPACE


def test_constraint_prefers_ari_only_among_configs_that_keep_transfer(tmp_path):
    """The point of the constraint: a big ARI win that costs transfer loses."""
    from lilyvae.search.ledger import Ledger, Trial
    from lilyvae.search.space import SearchConfig

    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append(Trial(
        index=0, config=SearchConfig().to_dict(),
        ari_median=0.23, auc_median=0.96,
        auc_transfer_median=0.806,
        auc_transfer_folds=[0.794, 0.806, 0.818],   # sd ~0.012
        proposed_by="baseline",
    ))
    # Best ARI by far, but transfer collapses.
    ledger.append(Trial(
        index=1, config=SearchConfig(latent_dim=6).to_dict(),
        ari_median=0.32, auc_median=0.95,
        auc_transfer_median=0.780, auc_transfer_folds=[0.78, 0.78, 0.78],
    ))
    # Smaller ARI gain, transfer held.
    ledger.append(Trial(
        index=2, config=SearchConfig(beta_end=0.3).to_dict(),
        ari_median=0.28, auc_median=0.95,
        auc_transfer_median=0.800, auc_transfer_folds=[0.80, 0.80, 0.80],
    ))

    assert ledger.best().index == 2, "a transfer-destroying config must not win"
    assert ledger.ranked()[-1].index == 1
    assert ledger.transfer_floor() < 0.806


def test_transfer_floor_is_permissive_without_a_baseline(tmp_path):
    """No baseline must leave everything admissible, not reject everything."""
    from lilyvae.search.ledger import Ledger, Trial
    from lilyvae.search.space import SearchConfig

    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append(Trial(index=0, config=SearchConfig().to_dict(),
                        ari_median=0.25, auc_median=0.95,
                        auc_transfer_median=0.70))
    assert ledger.transfer_floor() == float("-inf")
    assert ledger.best().index == 0


def test_fold_spread_falls_back_to_the_iqr(tmp_path):
    """Trials recorded before per-fold values were stored still rank."""
    from lilyvae.search.ledger import Trial
    from lilyvae.search.space import SearchConfig

    t = Trial(index=0, config=SearchConfig().to_dict(),
              ari_median=0.25, auc_median=0.95,
              auc_transfer_median=0.80,
              auc_transfer_iqr=[0.794, 0.810])
    assert abs(t.transfer_spread - (0.810 - 0.794) / 1.349) < 1e-9
