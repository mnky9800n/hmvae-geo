"""Regression tests against paper results.

These tests require preprocessed data in data/processed/ and trained
checkpoints in data/checkpoints/. They verify that the codebase still
produces the numbers reported in the paper. If any of these fail after
a refactor, the change broke something that matters.

Skip with: pytest -m "not requires_data"
Run slow tests: pytest -m slow
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from lilyvae.config import (
    PROCESSED_DIR, CHECKPOINT_DIR, BATCH_SIZE,
    INPUT_DIM, LATENT_DIM, NUM_LITHOLOGY_CLASSES,
)

# Skip all tests in this module if data is not present
pytestmark = pytest.mark.requires_data

def _data_available():
    return (
        (PROCESSED_DIR / "train_features.pt").exists()
        and (CHECKPOINT_DIR / "unsupervised_vae.pt").exists()
    )

if not _data_available():
    pytest.skip("Preprocessed data and checkpoints not found", allow_module_level=True)


# ---- Fixtures ----

@pytest.fixture(scope="module")
def labels():
    return torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)


@pytest.fixture(scope="module")
def features():
    return torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)


@pytest.fixture(scope="module")
def label_encoder():
    with open(PROCESSED_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def unsup_embeddings():
    return torch.load(CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True)


@pytest.fixture(scope="module")
def semi_embeddings():
    return torch.load(CHECKPOINT_DIR / "semi_supervised_embeddings.pt", weights_only=True)


@pytest.fixture(scope="module")
def dataloader():
    from lilyvae.data.dataset import get_dataloader
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture(scope="module")
def device():
    from lilyvae.training.trainer import get_device
    return get_device("auto")


def _load_vae(model_name):
    from lilyvae.model.vae import LilyVAE
    vae = LilyVAE()
    vae.load_state_dict(torch.load(
        CHECKPOINT_DIR / f"{model_name}_vae.pt",
        weights_only=True, map_location="cpu",
    ))
    return vae


# ---- Preprocessing determinism ----

class TestPreprocessing:
    def test_features_shape(self, features):
        assert features.shape == (238767, INPUT_DIM)

    def test_labels_shape(self, labels):
        assert labels.shape == (238767,)

    def test_num_classes(self, label_encoder):
        assert len(label_encoder.classes_) == NUM_LITHOLOGY_CLASSES

    def test_features_are_standardized(self, features):
        # After standardization, features should be roughly zero-mean unit-variance
        means = features.mean(dim=0)
        stds = features.std(dim=0)
        assert torch.allclose(means, torch.zeros(INPUT_DIM), atol=0.01)
        assert torch.allclose(stds, torch.ones(INPUT_DIM), atol=0.1)

    def test_labels_in_valid_range(self, labels):
        assert labels.min().item() >= 0
        assert labels.max().item() < NUM_LITHOLOGY_CLASSES

    def test_no_nan_in_features(self, features):
        assert not torch.isnan(features).any()

    def test_no_nan_in_labels(self, labels):
        assert not torch.isnan(labels.float()).any()

    def test_scaler_exists(self):
        assert (PROCESSED_DIR / "scaler.pkl").exists()

    def test_metadata_exists(self):
        metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
        assert "borehole_ids" in metadata
        assert len(metadata["borehole_ids"]) == 238767

    def test_zeroshot_targets_exist(self):
        targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
        assert isinstance(targets, dict)
        assert len(targets) > 50  # should have ~65 target variables


# ---- Checkpoint loading ----

class TestCheckpoints:
    def test_unsup_embeddings_shape(self, unsup_embeddings):
        assert unsup_embeddings.shape == (238767, LATENT_DIM)

    def test_semi_embeddings_shape(self, semi_embeddings):
        assert semi_embeddings.shape == (238767, LATENT_DIM)

    def test_unsup_checkpoint_loads(self):
        _load_vae("unsupervised")  # should not raise

    def test_semi_checkpoint_loads(self):
        _load_vae("semi_supervised")  # should not raise

    def test_semi_classifier_checkpoint_loads(self):
        from lilyvae.model.heads import LithologyClassifier
        clf = LithologyClassifier(LATENT_DIM, NUM_LITHOLOGY_CLASSES)
        clf.load_state_dict(torch.load(
            CHECKPOINT_DIR / "semi_supervised_classifier.pt",
            weights_only=True, map_location="cpu",
        ))

    def test_unsup_masking_checkpoint_loads(self):
        from lilyvae.training.masking import LearnableMasking
        masking = LearnableMasking()
        masking.load_state_dict(torch.load(
            CHECKPOINT_DIR / "unsupervised_masking.pt",
            weights_only=True, map_location="cpu",
        ))

    def test_unsup_embeddings_match_model(self, features, unsup_embeddings):
        vae = _load_vae("unsupervised")
        vae.eval()
        with torch.no_grad():
            z = vae.get_embeddings(features[:256])
        assert torch.allclose(z, unsup_embeddings[:256], atol=1e-5)

    def test_semi_embeddings_match_model(self, features, semi_embeddings):
        vae = _load_vae("semi_supervised")
        vae.eval()
        with torch.no_grad():
            z = vae.get_embeddings(features[:256])
        assert torch.allclose(z, semi_embeddings[:256], atol=1e-5)

    def test_embeddings_no_nan(self, unsup_embeddings, semi_embeddings):
        assert not torch.isnan(unsup_embeddings).any()
        assert not torch.isnan(semi_embeddings).any()

    def test_embeddings_finite(self, unsup_embeddings, semi_embeddings):
        assert torch.isfinite(unsup_embeddings).all()
        assert torch.isfinite(semi_embeddings).all()


# ---- Reconstruction R² ----

class TestReconstruction:
    def _load_and_eval(self, model_name, dataloader, device):
        from lilyvae.evaluation.reconstruction import evaluate_reconstruction
        vae = _load_vae(model_name)
        vae.to(device)
        return evaluate_reconstruction(vae, dataloader, device)

    def test_unsupervised_r2_avg(self, dataloader, device):
        # Paper reports median 0.997 [0.990, 0.999] from bootstrap; full-data
        # reconstruction should stay well above 0.99 regardless of training run.
        recon = self._load_and_eval("unsupervised", dataloader, device)
        assert recon["r2_avg"] > 0.99

    def test_semi_supervised_r2_avg(self, dataloader, device):
        # Paper reports median 0.997 [0.994, 0.999] from bootstrap; the
        # classification head shifts this slightly so we allow 0.97+.
        recon = self._load_and_eval("semi_supervised", dataloader, device)
        assert recon["r2_avg"] > 0.97

    def test_unsupervised_all_variables_above_threshold(self, dataloader, device):
        recon = self._load_and_eval("unsupervised", dataloader, device)
        for var in ["bulk_density", "magnetic_susceptibility", "NGR", "R", "G", "B"]:
            assert recon[f"r2_{var}"] > 0.95, f"Unsupervised R² for {var} dropped below 0.95"

    def test_semi_supervised_all_variables_above_threshold(self, dataloader, device):
        recon = self._load_and_eval("semi_supervised", dataloader, device)
        for var in ["bulk_density", "magnetic_susceptibility", "NGR", "R", "G", "B"]:
            assert recon[f"r2_{var}"] > 0.93, f"Semi-supervised R² for {var} dropped below 0.93"

    def test_reconstruction_returns_all_keys(self, dataloader, device):
        recon = self._load_and_eval("unsupervised", dataloader, device)
        expected = {"r2_avg", "r2_bulk_density", "r2_magnetic_susceptibility",
                    "r2_NGR", "r2_R", "r2_G", "r2_B"}
        assert expected == set(recon.keys())


# ---- Classification AUC ----

class TestClassification:
    # Paper reports AUC 0.957 (unsup) and 0.956 (semi) as bootstrap median.
    # Full-data single-point evaluation typically falls in 0.94-0.96;
    # cross-machine variation from different training runs stays within
    # a few tenths of a percent.
    def test_unsupervised_auc(self, unsup_embeddings, labels):
        from lilyvae.evaluation.classification import evaluate_classification
        result = evaluate_classification(unsup_embeddings, labels)
        assert result["auc_macro"] > 0.93

    def test_semi_supervised_auc(self, semi_embeddings, labels):
        from lilyvae.evaluation.classification import evaluate_classification
        result = evaluate_classification(semi_embeddings, labels)
        assert result["auc_macro"] > 0.93

    def test_num_classes_evaluated(self, unsup_embeddings, labels):
        from lilyvae.evaluation.classification import evaluate_classification
        result = evaluate_classification(unsup_embeddings, labels)
        assert result["n_classes_evaluated"] == 134

    def test_all_per_class_aucs_valid(self, unsup_embeddings, labels):
        from lilyvae.evaluation.classification import evaluate_classification
        result = evaluate_classification(unsup_embeddings, labels)
        for cls, auc in result["per_class_auc"].items():
            assert 0.5 <= auc <= 1.0, f"Class {cls} has invalid AUC {auc}"


# ---- Clustering ARI ----

class TestClustering:
    # Paper reports ARI 0.257 (unsup) and 0.256 (semi) as bootstrap median.
    # Full-data single-point ARI is lower (typically 0.11-0.15) because
    # GMM at k=10 partitions all 139 classes instead of a held-out subset.
    # Cross-run variation from training nondeterminism can shift this by
    # ~0.02 in either direction, so we test a loose sanity threshold.
    def test_unsupervised_ari(self, unsup_embeddings, labels):
        from lilyvae.evaluation.clustering import evaluate_clustering
        result = evaluate_clustering(unsup_embeddings, labels)
        assert result["ari"] > 0.08

    def test_semi_supervised_ari(self, semi_embeddings, labels):
        from lilyvae.evaluation.clustering import evaluate_clustering
        result = evaluate_clustering(semi_embeddings, labels)
        assert result["ari"] > 0.08

    def test_clustering_uses_10_components(self, unsup_embeddings, labels):
        from lilyvae.evaluation.clustering import evaluate_clustering
        result = evaluate_clustering(unsup_embeddings, labels)
        assert result["n_components"] == 10


# ---- Zero-shot (slow) ----

@pytest.mark.slow
class TestZeroShot:
    @pytest.fixture(scope="class")
    def zeroshot_targets(self):
        return torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    @pytest.fixture(scope="class")
    def unsup_zeroshot_results(self, unsup_embeddings, zeroshot_targets):
        from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot
        return evaluate_all_zeroshot(unsup_embeddings, zeroshot_targets)

    def test_thermal_conductivity(self, unsup_zeroshot_results):
        r2 = unsup_zeroshot_results["TCON_Thermal conductivity mean (W/(m*K))"]["r2"]
        assert r2 > 0.99

    def test_porosity(self, unsup_zeroshot_results):
        r2 = unsup_zeroshot_results["MAD_Porosity (vol%)"]["r2"]
        assert r2 > 0.80

    def test_pwc_velocity(self, unsup_zeroshot_results):
        r2 = unsup_zeroshot_results["PWC_P-wave velocity x (m/s)"]["r2"]
        assert r2 > 0.99

    def test_reflectance_l_star(self, unsup_zeroshot_results):
        r2 = unsup_zeroshot_results["RSC_Reflectance L* (234766 samples)".replace(" (234766 samples)", "")]["r2"]
        # Use the actual key
        matching = [k for k in unsup_zeroshot_results if "Reflectance L*" in k]
        assert len(matching) == 1
        assert unsup_zeroshot_results[matching[0]]["r2"] > 0.90

    def test_magnetic_declination_unpredictable(self, unsup_zeroshot_results):
        matching = [k for k in unsup_zeroshot_results if k == "SRM_Declination raw (deg)"]
        assert len(matching) == 1
        r2 = unsup_zeroshot_results[matching[0]]["r2"]
        assert r2 < 0.15  # should be ~0.09

    def test_num_zeroshot_variables(self, unsup_zeroshot_results):
        # We evaluate ~65 variables; make sure none disappeared
        valid = [k for k, v in unsup_zeroshot_results.items()
                 if not np.isnan(v.get("r2", float("nan")))]
        assert len(valid) >= 60
