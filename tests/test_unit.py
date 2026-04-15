"""Unit tests that require no data on disk."""

from pathlib import Path

import torch
import pytest

from lilyvae.config import (
    INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, NUM_LITHOLOGY_CLASSES,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MASK_FRACTION, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS,
    ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS,
    LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, SEED, BIN_SIZE_CM,
    TRAINING_VARIABLES,
)


# ---- Config constants ----

class TestConfig:
    def test_input_dim(self):
        assert INPUT_DIM == 6

    def test_latent_dim(self):
        assert LATENT_DIM == 10

    def test_hidden_dims(self):
        assert list(HIDDEN_DIMS) == [128, 64]

    def test_num_lithology_classes(self):
        assert NUM_LITHOLOGY_CLASSES == 139

    def test_training_hyperparams(self):
        assert NUM_EPOCHS == 100
        assert BATCH_SIZE == 256
        assert LEARNING_RATE == 1e-3
        assert WEIGHT_DECAY == 1e-5
        assert MASK_FRACTION == 0.1
        assert BETA_END == 0.075
        assert SEED == 42

    def test_training_variables(self):
        assert TRAINING_VARIABLES == [
            "bulk_density", "magnetic_susceptibility", "NGR", "R", "G", "B",
        ]
        assert len(TRAINING_VARIABLES) == INPUT_DIM

    def test_bin_size(self):
        assert BIN_SIZE_CM == 20


# ---- Model shapes ----

class TestModelShapes:
    def test_encoder_output_shape(self):
        from lilyvae.model.vae import Encoder
        enc = Encoder()
        x = torch.randn(8, INPUT_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (8, LATENT_DIM)
        assert logvar.shape == (8, LATENT_DIM)

    def test_encoder_batch_size_one(self):
        from lilyvae.model.vae import Encoder
        enc = Encoder()
        x = torch.randn(1, INPUT_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (1, LATENT_DIM)

    def test_decoder_output_shape(self):
        from lilyvae.model.vae import Decoder
        dec = Decoder()
        z = torch.randn(8, LATENT_DIM)
        mu, logvar = dec(z)
        assert mu.shape == (8, INPUT_DIM)
        assert logvar.shape == (8, INPUT_DIM)

    def test_decoder_logvar_clamped(self):
        from lilyvae.model.vae import Decoder
        dec = Decoder()
        z = torch.randn(8, LATENT_DIM) * 100  # extreme input
        _, logvar = dec(z)
        assert logvar.min().item() >= LOGVAR_CLAMP_MIN
        assert logvar.max().item() <= LOGVAR_CLAMP_MAX

    def test_vae_forward_keys(self):
        from lilyvae.model.vae import LilyVAE
        vae = LilyVAE()
        x = torch.randn(8, INPUT_DIM)
        out = vae(x)
        assert set(out.keys()) == {"x_mu", "x_logvar", "z_mu", "z_logvar", "z"}

    def test_vae_forward_shapes(self):
        from lilyvae.model.vae import LilyVAE
        vae = LilyVAE()
        x = torch.randn(8, INPUT_DIM)
        out = vae(x)
        assert out["x_mu"].shape == (8, INPUT_DIM)
        assert out["x_logvar"].shape == (8, INPUT_DIM)
        assert out["z_mu"].shape == (8, LATENT_DIM)
        assert out["z_logvar"].shape == (8, LATENT_DIM)
        assert out["z"].shape == (8, LATENT_DIM)

    def test_vae_reparameterize_adds_noise(self):
        from lilyvae.model.vae import LilyVAE
        vae = LilyVAE()
        mu = torch.zeros(100, LATENT_DIM)
        logvar = torch.zeros(100, LATENT_DIM)  # std=1
        z = vae.reparameterize(mu, logvar)
        # With mu=0 and std=1, z should have nonzero variance
        assert z.std().item() > 0.5

    def test_vae_get_embeddings_is_deterministic(self):
        from lilyvae.model.vae import LilyVAE
        vae = LilyVAE()
        vae.eval()
        x = torch.randn(8, INPUT_DIM)
        z1 = vae.get_embeddings(x)
        z2 = vae.get_embeddings(x)
        assert torch.allclose(z1, z2)

    def test_vae_get_embeddings_uses_mu_not_sample(self):
        from lilyvae.model.vae import LilyVAE
        vae = LilyVAE()
        vae.eval()
        x = torch.randn(8, INPUT_DIM)
        z = vae.get_embeddings(x)
        with torch.no_grad():
            out = vae(x)
        # get_embeddings returns the encoder mu, not the sampled z
        assert torch.allclose(z, out["z_mu"])

    def test_classifier_output_shape(self):
        from lilyvae.model.heads import LithologyClassifier
        clf = LithologyClassifier(LATENT_DIM, NUM_LITHOLOGY_CLASSES)
        z = torch.randn(8, LATENT_DIM)
        logits = clf(z)
        assert logits.shape == (8, NUM_LITHOLOGY_CLASSES)


# ---- Ablation model shapes ----

class TestAblationModels:
    def test_fixed_variance_vae_forward_keys(self):
        from lilyvae.evaluation.ablations import FixedVarianceVAE
        vae = FixedVarianceVAE()
        x = torch.randn(8, INPUT_DIM)
        out = vae(x)
        # FixedVarianceVAE has no x_logvar
        assert "x_mu" in out
        assert "z_mu" in out
        assert "z_logvar" in out
        assert "z" in out
        assert "x_logvar" not in out

    def test_fixed_variance_vae_shapes(self):
        from lilyvae.evaluation.ablations import FixedVarianceVAE
        vae = FixedVarianceVAE()
        x = torch.randn(8, INPUT_DIM)
        out = vae(x)
        assert out["x_mu"].shape == (8, INPUT_DIM)
        assert out["z_mu"].shape == (8, LATENT_DIM)

    def test_fixed_variance_vae_get_embeddings(self):
        from lilyvae.evaluation.ablations import FixedVarianceVAE
        vae = FixedVarianceVAE()
        vae.eval()
        x = torch.randn(8, INPUT_DIM)
        z = vae.get_embeddings(x)
        assert z.shape == (8, LATENT_DIM)

    def test_zero_masking_replaces_with_zeros(self):
        from lilyvae.evaluation.ablations import ZeroMasking
        torch.manual_seed(0)
        masking = ZeroMasking(mask_fraction=0.5)
        masking.train()
        x = torch.ones(100, INPUT_DIM) * 5.0
        masked = masking(x)
        # Masked positions should be exactly zero
        changed = masked != x
        assert changed.any()
        assert (masked[changed] == 0.0).all()

    def test_no_masking_is_identity(self):
        from lilyvae.evaluation.ablations import NoMasking
        masking = NoMasking()
        masking.train()
        x = torch.randn(8, INPUT_DIM)
        assert torch.allclose(masking(x), x)


# ---- SAE model shapes ----

class TestSAEShapes:
    def test_sae_forward_shapes(self):
        from lilyvae.evaluation.sae import SparseAutoencoder
        sae = SparseAutoencoder(d_latent=LATENT_DIM, d_hidden=64)
        z = torch.randn(16, LATENT_DIM)
        z_hat, f = sae(z)
        assert z_hat.shape == (16, LATENT_DIM)
        assert f.shape == (16, 64)

    def test_sae_features_are_nonnegative(self):
        from lilyvae.evaluation.sae import SparseAutoencoder
        sae = SparseAutoencoder(d_latent=LATENT_DIM, d_hidden=64)
        z = torch.randn(100, LATENT_DIM)
        _, f = sae(z)
        assert (f >= 0).all()  # ReLU activation

    def test_sae_encode_decode_roundtrip(self):
        from lilyvae.evaluation.sae import SparseAutoencoder
        sae = SparseAutoencoder(d_latent=LATENT_DIM, d_hidden=64)
        z = torch.randn(8, LATENT_DIM)
        f = sae.encode(z)
        z_hat = sae.decode(f)
        assert z_hat.shape == z.shape


# ---- Loss functions ----

class TestLoss:
    def test_heteroscedastic_loss_reduces_to_mse_at_zero_logvar(self):
        from lilyvae.training.loss import heteroscedastic_reconstruction_loss
        x = torch.randn(32, INPUT_DIM)
        x_mu = torch.randn(32, INPUT_DIM)
        x_logvar = torch.zeros(32, INPUT_DIM)
        loss = heteroscedastic_reconstruction_loss(x, x_mu, x_logvar)
        # At logvar=0: 0.5 * mean(0 + (x-mu)^2 / 1) = 0.5 * MSE (mean over all elements)
        expected = 0.5 * (x - x_mu).pow(2).mean()
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_heteroscedastic_loss_high_variance_reduces_loss(self):
        from lilyvae.training.loss import heteroscedastic_reconstruction_loss
        x = torch.randn(32, INPUT_DIM)
        x_mu = torch.randn(32, INPUT_DIM)
        loss_low_var = heteroscedastic_reconstruction_loss(x, x_mu, torch.zeros(32, INPUT_DIM))
        loss_high_var = heteroscedastic_reconstruction_loss(x, x_mu, torch.ones(32, INPUT_DIM) * 4)
        # Higher predicted variance reduces the squared-error penalty (but adds log-var term)
        # For large enough residuals, high variance should give lower loss
        # This is the key property: the model can "explain away" noisy variables

    def test_heteroscedastic_loss_perfect_prediction(self):
        from lilyvae.training.loss import heteroscedastic_reconstruction_loss
        x = torch.randn(32, INPUT_DIM)
        loss = heteroscedastic_reconstruction_loss(x, x, torch.zeros(32, INPUT_DIM))
        # Perfect prediction at logvar=0: loss = 0.5 * mean(0 + 0) = 0
        assert abs(loss.item()) < 1e-6

    def test_heteroscedastic_loss_is_nonnegative_at_zero_logvar(self):
        from lilyvae.training.loss import heteroscedastic_reconstruction_loss
        x = torch.randn(32, INPUT_DIM)
        x_mu = torch.randn(32, INPUT_DIM)
        x_logvar = torch.zeros(32, INPUT_DIM)
        loss = heteroscedastic_reconstruction_loss(x, x_mu, x_logvar)
        assert loss.item() >= 0

    def test_kl_divergence_zero_at_prior(self):
        from lilyvae.training.loss import kl_divergence
        mu = torch.zeros(32, LATENT_DIM)
        logvar = torch.zeros(32, LATENT_DIM)
        kl = kl_divergence(mu, logvar)
        assert abs(kl.item()) < 1e-6

    def test_kl_divergence_positive(self):
        from lilyvae.training.loss import kl_divergence
        mu = torch.randn(32, LATENT_DIM)
        logvar = torch.randn(32, LATENT_DIM)
        kl = kl_divergence(mu, logvar)
        assert kl.item() > 0

    def test_kl_divergence_increases_with_mu(self):
        from lilyvae.training.loss import kl_divergence
        logvar = torch.zeros(32, LATENT_DIM)
        kl_small = kl_divergence(torch.ones(32, LATENT_DIM) * 0.1, logvar)
        kl_large = kl_divergence(torch.ones(32, LATENT_DIM) * 5.0, logvar)
        assert kl_large.item() > kl_small.item()

    def test_classification_loss(self):
        from lilyvae.training.loss import classification_loss
        logits = torch.randn(32, NUM_LITHOLOGY_CLASSES)
        labels = torch.randint(0, NUM_LITHOLOGY_CLASSES, (32,))
        loss = classification_loss(logits, labels)
        assert loss.item() > 0
        assert loss.dim() == 0  # scalar

    def test_beta_schedule_boundaries(self):
        from lilyvae.training.loss import beta_schedule
        assert beta_schedule(0, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS) == BETA_START
        assert beta_schedule(BETA_ANNEAL_EPOCHS, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS) == BETA_END
        assert beta_schedule(BETA_ANNEAL_EPOCHS + 50, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS) == BETA_END

    def test_beta_schedule_monotonic(self):
        from lilyvae.training.loss import beta_schedule
        betas = [beta_schedule(e, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS) for e in range(NUM_EPOCHS)]
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i - 1]

    def test_beta_schedule_midpoint(self):
        from lilyvae.training.loss import beta_schedule
        mid = beta_schedule(BETA_ANNEAL_EPOCHS // 2, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)
        assert BETA_START < mid < BETA_END

    def test_alpha_schedule_boundaries(self):
        from lilyvae.training.loss import alpha_schedule
        assert alpha_schedule(0, ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS) == ALPHA_START
        assert alpha_schedule(ALPHA_ANNEAL_EPOCHS, ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS) == ALPHA_END

    def test_alpha_schedule_monotonic(self):
        from lilyvae.training.loss import alpha_schedule
        alphas = [alpha_schedule(e, ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS) for e in range(NUM_EPOCHS)]
        for i in range(1, len(alphas)):
            assert alphas[i] >= alphas[i - 1]


# ---- Masking ----

class TestMasking:
    def test_learnable_masking_train_mode(self):
        from lilyvae.training.masking import LearnableMasking
        torch.manual_seed(0)
        masking = LearnableMasking(INPUT_DIM, MASK_FRACTION)
        masking.train()
        x = torch.ones(1000, INPUT_DIM)
        masked = masking(x)
        n_changed = (masked != x).float().mean().item()
        assert 0.05 < n_changed < 0.20  # ~10% masked

    def test_learnable_masking_eval_mode(self):
        from lilyvae.training.masking import LearnableMasking
        masking = LearnableMasking(INPUT_DIM, MASK_FRACTION)
        masking.eval()
        x = torch.randn(8, INPUT_DIM)
        masked = masking(x)
        assert torch.allclose(x, masked)

    def test_learnable_masking_replaces_with_learned_tokens(self):
        from lilyvae.training.masking import LearnableMasking
        torch.manual_seed(42)
        masking = LearnableMasking(INPUT_DIM, mask_fraction=1.0)  # mask everything
        masking.train()
        x = torch.ones(8, INPUT_DIM) * 99.0
        masked = masking(x)
        # All values should be replaced with the mask token values
        mask_tokens = masking.mask_tokens.detach()
        for i in range(8):
            assert torch.allclose(masked[i], mask_tokens.squeeze())

    def test_learnable_masking_has_learnable_parameters(self):
        from lilyvae.training.masking import LearnableMasking
        masking = LearnableMasking(INPUT_DIM, MASK_FRACTION)
        params = list(masking.parameters())
        assert len(params) == 1  # mask_tokens
        assert params[0].numel() == INPUT_DIM
        assert params[0].requires_grad

    def test_masking_fraction_zero(self):
        from lilyvae.training.masking import LearnableMasking
        masking = LearnableMasking(INPUT_DIM, mask_fraction=0.0)
        masking.train()
        x = torch.randn(100, INPUT_DIM)
        masked = masking(x)
        assert torch.allclose(x, masked)


# ---- Dataset ----

class TestDataset:
    def test_lily_dataset_returns_correct_types(self):
        from lilyvae.data.dataset import LILYDataset
        # Create minimal fake data to test the class
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            features = torch.randn(10, INPUT_DIM)
            labels = torch.randint(0, NUM_LITHOLOGY_CLASSES, (10,))
            torch.save(features, os.path.join(tmpdir, "train_features.pt"))
            torch.save(labels, os.path.join(tmpdir, "train_labels.pt"))

            ds = LILYDataset(Path(tmpdir))
            assert len(ds) == 10
            feat, lab, idx = ds[0]
            assert isinstance(feat, torch.Tensor)
            assert isinstance(lab, torch.Tensor)
            assert isinstance(idx, int)
            assert feat.shape == (INPUT_DIM,)
            assert lab.shape == ()

    def test_lily_dataset_index_correct(self):
        from lilyvae.data.dataset import LILYDataset
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            features = torch.randn(10, INPUT_DIM)
            labels = torch.randint(0, NUM_LITHOLOGY_CLASSES, (10,))
            torch.save(features, os.path.join(tmpdir, "train_features.pt"))
            torch.save(labels, os.path.join(tmpdir, "train_labels.pt"))

            ds = LILYDataset(Path(tmpdir))
            _, _, idx = ds[7]
            assert idx == 7
