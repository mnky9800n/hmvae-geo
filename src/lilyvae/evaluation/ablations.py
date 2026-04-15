"""Ablation model components for LilyVAE.

Provides variant model classes and a training loop used by the bootstrap
evaluator to train ablation models (one architectural change at a time).

- FixedVarianceVAE: MSE decoder instead of heteroscedastic Gaussian.
- ZeroMasking: replaces masked features with zero instead of learned tokens.
- NoMasking: no masking at all.
- _train_ablation_model: training loop that handles all variant configs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lilyvae.config import (
    INPUT_DIM,
    LATENT_DIM,
    HIDDEN_DIMS,
    MASK_FRACTION,
    BETA_START,
    BETA_END,
    BETA_ANNEAL_EPOCHS,
    SEED,
)
from lilyvae.model.vae import LilyVAE, Encoder
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.loss import (
    heteroscedastic_reconstruction_loss,
    kl_divergence,
    beta_schedule,
)
from lilyvae.training.trainer import get_device


# ---------------------------------------------------------------------------
# Variant model components
# ---------------------------------------------------------------------------


class FixedVarianceDecoder(nn.Module):
    """Standard decoder that outputs only a mean (no per-variable log-variance).

    Uses the same hidden architecture as the heteroscedastic Decoder but
    exposes a single output head. Returns a dict with "x_mu" so that it
    drops into evaluate_reconstruction without modification.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dims: list[int] | None = None,
        output_dim: int = INPUT_DIM,
    ):
        super().__init__()
        hidden_dims = list(reversed(hidden_dims or list(HIDDEN_DIMS)))

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.network(z))


class FixedVarianceVAE(nn.Module):
    """VAE with a fixed-variance (MSE) decoder.

    Structurally identical to LilyVAE except the decoder outputs only x_mu
    and reconstruction is measured with MSE. The forward output dict matches
    the shape that evaluate_reconstruction and evaluate_clustering expect.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = FixedVarianceDecoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decoder(z)
        return {
            "x_mu": x_mu,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
        }

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, _ = self.encoder(x)
        return z_mu


# ---------------------------------------------------------------------------
# Masking variants
# ---------------------------------------------------------------------------


class ZeroMasking(nn.Module):
    """Masks features by replacing them with zero (no learnable tokens).

    After standardization, zero equals the training-set mean, which is
    distinct from a learned token but ambiguous with in-distribution values.
    This serves as the ablation of the learnable-token design choice.
    """

    def __init__(self, mask_fraction: float = MASK_FRACTION):
        super().__init__()
        self.mask_fraction = mask_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mask = torch.rand_like(x) < self.mask_fraction
        return torch.where(mask, torch.zeros_like(x), x)


class NoMasking(nn.Module):
    """Pass-through: applies no masking at all during training."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_masking(masking_type: str, mask_fraction: float) -> nn.Module:
    if masking_type == "learnable":
        return LearnableMasking(INPUT_DIM, mask_fraction)
    if masking_type == "zero":
        return ZeroMasking(mask_fraction)
    if masking_type == "none":
        return NoMasking()
    raise ValueError(f"Unknown masking_type: {masking_type!r}")


def _make_vae(decoder_type: str, latent_dim: int, hidden_dims: list[int]) -> nn.Module:
    if decoder_type == "heteroscedastic":
        return LilyVAE(INPUT_DIM, latent_dim, hidden_dims)
    if decoder_type == "fixed_variance":
        return FixedVarianceVAE(INPUT_DIM, latent_dim, hidden_dims)
    raise ValueError(f"Unknown decoder_type: {decoder_type!r}")


# ---------------------------------------------------------------------------
# Training loop for ablations
# ---------------------------------------------------------------------------


def _train_ablation_model(
    dataloader: DataLoader,
    vae: nn.Module,
    masking: nn.Module,
    decoder_type: str,
    beta_schedule_type: str,
    alpha_weight: float,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> nn.Module:
    """Train vae + masking in-place and return the trained vae.

    When alpha_weight > 0, a classification head is added for lithology
    (semi-supervised training). alpha anneals from 0 to alpha_weight
    over the first 50 epochs.
    """
    from lilyvae.model.heads import LithologyClassifier
    from lilyvae.config import NUM_LITHOLOGY_CLASSES, ALPHA_ANNEAL_EPOCHS

    vae = vae.to(device)
    masking = masking.to(device)
    params = list(vae.parameters()) + list(masking.parameters())

    classifier = None
    if alpha_weight > 0:
        latent_dim = vae.encoder.fc_mu.out_features
        classifier = LithologyClassifier(latent_dim, NUM_LITHOLOGY_CLASSES).to(device)
        params += list(classifier.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    fixed_beta = BETA_END

    for epoch in range(num_epochs):
        vae.train()
        masking.train()
        if classifier:
            classifier.train()

        if beta_schedule_type == "annealing":
            beta = beta_schedule(epoch, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)
        else:
            beta = fixed_beta

        # Alpha annealing: 0 -> alpha_weight over ALPHA_ANNEAL_EPOCHS
        if classifier:
            if epoch >= ALPHA_ANNEAL_EPOCHS:
                alpha = alpha_weight
            else:
                alpha = alpha_weight * (epoch / ALPHA_ANNEAL_EPOCHS)
        else:
            alpha = 0.0

        for features, labels, _ in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            masked = masking(features)
            out = vae(masked)

            if decoder_type == "heteroscedastic":
                recon = heteroscedastic_reconstruction_loss(
                    features, out["x_mu"], out["x_logvar"]
                )
            else:
                recon = F.mse_loss(out["x_mu"], features)

            kl = kl_divergence(out["z_mu"], out["z_logvar"])
            loss = recon + beta * kl

            if classifier:
                logits = classifier(out["z"])
                cls_loss = F.cross_entropy(logits, labels)
                loss = loss + alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"  epoch {epoch+1:3d}/{num_epochs} | beta={beta:.6f}"
            if classifier:
                msg += f" | alpha={alpha:.6f}"
            print(msg)

    return vae


