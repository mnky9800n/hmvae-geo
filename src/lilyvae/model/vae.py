"""Variational autoencoder with heteroscedastic decoder."""

import torch
import torch.nn as nn

from lilyvae.config import (
    INPUT_DIM,
    LATENT_DIM,
    HIDDEN_DIMS,
    LOGVAR_CLAMP_MIN,
    LOGVAR_CLAMP_MAX,
)


class Encoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dims: list[int] = None, latent_dim: int = LATENT_DIM):
        super().__init__()
        hidden_dims = hidden_dims or list(HIDDEN_DIMS)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.network = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Heteroscedastic Gaussian decoder: outputs per-variable mean and log-variance."""

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dims: list[int] = None, output_dim: int = INPUT_DIM):
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
        self.fc_logvar = nn.Linear(in_dim, output_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(z)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)
        return mu, logvar


class LilyVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decoder(z)
        return {
            "x_mu": x_mu,
            "x_logvar": x_logvar,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
        }

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic embeddings (encoder mu, no sampling)."""
        z_mu, _ = self.encoder(x)
        return z_mu
