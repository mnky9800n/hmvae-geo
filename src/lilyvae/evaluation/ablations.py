"""Ablation study framework for LilyVAE.

Each ablation trains a fresh model variant with one hyperparameter or
architectural choice changed, then evaluates the same three metrics used
throughout the project: reconstruction R², ARI (k=10 GMM), and macro AUC.

Design notes
------------
- Existing classes (LilyVAE, LearnableMasking) are never modified.
- Variants are expressed through factory functions that produce objects
  compatible with the training loop's expectations.
- "fixed_variance" decoder: a thin FixedVarianceVAE wrapper that outputs
  x_mu only, paired with MSE reconstruction loss. The output dict shape
  matches what evaluate_reconstruction reads (it only uses "x_mu").
- Masking variants: ZeroMasking and NoMasking are minimal classes that
  match the LearnableMasking interface (nn.Module, training-aware forward).
- Each ablation is a named configuration; run_ablation compiles that
  configuration into a trained model and returns metrics.
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
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MASK_FRACTION,
    BETA_START,
    BETA_END,
    BETA_ANNEAL_EPOCHS,
    PROCESSED_DIR,
    BATCH_SIZE,
    SEED,
)
from lilyvae.model.vae import LilyVAE, Encoder, Decoder
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.loss import (
    heteroscedastic_reconstruction_loss,
    kl_divergence,
    beta_schedule,
)
from lilyvae.training.trainer import get_device
from lilyvae.data.dataset import get_dataloader
from lilyvae.evaluation.reconstruction import evaluate_reconstruction
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.classification import evaluate_classification


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
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> nn.Module:
    """Train vae + masking in-place and return the trained vae.

    Kept private: callers should use run_ablation or run_all_ablations.
    """
    vae = vae.to(device)
    masking = masking.to(device)
    params = list(vae.parameters()) + list(masking.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    fixed_beta = BETA_END  # used when beta_schedule_type == "fixed"

    for epoch in range(num_epochs):
        vae.train()
        masking.train()

        if beta_schedule_type == "annealing":
            beta = beta_schedule(epoch, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)
        else:
            beta = fixed_beta

        for features, _, _ in dataloader:
            features = features.to(device)
            masked = masking(features)
            out = vae(masked)

            if decoder_type == "heteroscedastic":
                recon = heteroscedastic_reconstruction_loss(
                    features, out["x_mu"], out["x_logvar"]
                )
            else:
                # fixed_variance: plain MSE averaged over all elements
                recon = F.mse_loss(out["x_mu"], features)

            kl = kl_divergence(out["z_mu"], out["z_logvar"])
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d}/{num_epochs} | beta={beta:.6f}"
            )

    return vae


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: All supported ablation names and their candidate values.
ABLATION_GRID: dict[str, list] = {
    "decoder_type": ["heteroscedastic", "fixed_variance"],
    "masking_type": ["learnable", "zero", "none"],
    "mask_fraction": [0.0, 0.1, 0.2, 0.3],
    "latent_dim": [2, 5, 10, 20, 50],
    "beta_schedule": ["annealing", "fixed"],
}


def run_ablation(
    ablation_name: str,
    value,
    *,
    dataloader: DataLoader | None = None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    device: str = "auto",
) -> dict:
    """Train and evaluate one ablation variant.

    Args:
        ablation_name: One of the keys in ABLATION_GRID.
        value: The value for that ablation (must be in ABLATION_GRID[ablation_name]).
        dataloader: Optional pre-built DataLoader. If None, builds one from
            PROCESSED_DIR (shuffle=False so embeddings stay aligned with labels).
        num_epochs: Training epochs for this run.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Gradient clipping threshold.
        device: "auto", "cuda", "mps", or "cpu".

    Returns:
        Dict with keys: ablation_name, value, r2_avg, ari, auc_macro,
        and the full per-variable r2 breakdown.
    """
    if ablation_name not in ABLATION_GRID:
        raise ValueError(
            f"Unknown ablation {ablation_name!r}. "
            f"Choose from: {list(ABLATION_GRID)}"
        )

    torch.manual_seed(SEED)
    dev = get_device(device)

    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    # Resolve configuration, filling in defaults for everything not ablated.
    decoder_type = value if ablation_name == "decoder_type" else "heteroscedastic"
    masking_type = value if ablation_name == "masking_type" else "learnable"
    mask_fraction = value if ablation_name == "mask_fraction" else MASK_FRACTION
    latent_dim = value if ablation_name == "latent_dim" else LATENT_DIM
    beta_schedule_type = value if ablation_name == "beta_schedule" else "annealing"
    hidden_dims = list(HIDDEN_DIMS)

    # masking_type "none" means mask_fraction is irrelevant; zero still valid
    vae = _make_vae(decoder_type, latent_dim, hidden_dims)
    masking = _make_masking(masking_type, mask_fraction)

    print(
        f"\n[ablation] {ablation_name}={value!r}  "
        f"(decoder={decoder_type}, masking={masking_type}, "
        f"frac={mask_fraction}, latent_dim={latent_dim}, "
        f"beta_schedule={beta_schedule_type})"
    )

    vae = _train_ablation_model(
        dataloader=dataloader,
        vae=vae,
        masking=masking,
        decoder_type=decoder_type,
        beta_schedule_type=beta_schedule_type,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=dev,
    )

    # Evaluation — use a non-shuffled loader so embeddings align with labels.
    eval_loader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)
    labels = _collect_labels(eval_loader)

    recon_metrics = evaluate_reconstruction(vae, eval_loader, dev)

    embeddings = _extract_embeddings(vae, eval_loader, dev)
    clust_metrics = evaluate_clustering(embeddings, labels)
    cls_metrics = evaluate_classification(embeddings, labels)

    return {
        "ablation_name": ablation_name,
        "value": value,
        **recon_metrics,
        "ari": clust_metrics["ari"],
        "auc_macro": cls_metrics["auc_macro"],
    }


def run_all_ablations(
    *,
    num_epochs: int = NUM_EPOCHS,
    device: str = "auto",
) -> dict[str, list[dict]]:
    """Run every value for every ablation and return all results.

    Results are keyed by ablation_name, each value being a list of per-value
    result dicts (as returned by run_ablation).

    Builds a single DataLoader upfront and reuses it across all runs to avoid
    redundant disk I/O on large datasets.
    """
    # shuffle=False: embeddings and labels stay aligned during evaluation
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)
    all_results: dict[str, list[dict]] = {}

    for ablation_name, values in ABLATION_GRID.items():
        all_results[ablation_name] = []
        for value in values:
            result = run_ablation(
                ablation_name,
                value,
                dataloader=dataloader,
                num_epochs=num_epochs,
                device=device,
            )
            all_results[ablation_name].append(result)

    return all_results


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _collect_labels(dataloader: DataLoader) -> torch.Tensor:
    all_labels = []
    for _, labels, _ in dataloader:
        all_labels.append(labels)
    return torch.cat(all_labels, dim=0)


def _extract_embeddings(
    vae: nn.Module, dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    vae.eval()
    all_z = []
    with torch.no_grad():
        for features, _, _ in dataloader:
            features = features.to(device)
            z_mu = vae.get_embeddings(features)
            all_z.append(z_mu.cpu())
    return torch.cat(all_z, dim=0)
