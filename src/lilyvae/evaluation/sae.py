"""Sparse autoencoder on the LilyVAE latent space for interpretability.

Trains a small sparse autoencoder on the 10-dim VAE latent representation
and checks, for each SAE feature, whether it matches any of four concrete
geological hypotheses the paper cares about:

1. Depth / compaction -- a feature whose activation correlates strongly
   with burial depth, indicating the VAE has implicitly learned depth
   from density and color gradients.

2. Carbonate-vs-clay -- a feature that discriminates carbonate ooze
   (nannofossil ooze, foraminifera ooze, chalk, etc.) from clay / mud.

3. Volcanic ash / tephra -- a feature that fires selectively on samples
   labeled as ash, tuff, tephra, or volcaniclastic rock.

4. Organic-rich -- a feature that fires on samples with high NGR (a proxy
   for uranium accumulation in organic-rich sediments) and/or on diatom
   ooze / diatomite.

The module exposes:
- `SparseAutoencoder`: a small nn.Module with a wide hidden layer and an
  L1 sparsity penalty on the hidden activations.
- `train_sae`: trains it on a precomputed stack of VAE latent vectors.
- `score_features_against_targets`: per-feature summary table (top activating
  samples, depth correlation, lithology fractions).
- `steer`: amplifies a single feature in a batch of latent vectors and
  returns the modified latents, which can then be decoded through the
  VAE decoder to see whether the intervention moves the reconstructed
  measurements in the expected direction.

The 10-dim latent is unusually small for SAE work (LLM residual-stream
SAEs operate on 768-8192 dims). We expand to a much wider hidden layer
but accept that if the latent has fewer than 10 independent directions
of variation, the SAE will just learn aliased copies of the latent axes
and nothing interesting will emerge. This module is a hypothesis test,
not a guarantee.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lilyvae.config import LATENT_DIM, SEED


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    """Tied-input-dim sparse autoencoder.

    Architecture:
        input (d_latent) -> Linear(d_latent, d_hidden) -> ReLU -> features
        features -> Linear(d_hidden, d_latent) -> reconstruction

    The encoder bias is the only centering applied; the user is expected
    to pass in standardized (zero-mean, unit-variance) latent vectors so
    that the L1 penalty scales comparably across dimensions.

    Loss is reconstruction MSE plus `l1_coeff * mean(ReLU(features))`.
    """

    def __init__(
        self,
        d_latent: int = LATENT_DIM,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(d_latent, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_latent, bias=False)

        # Initialize decoder to tied transpose of encoder (standard SAE init).
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Return the sparse feature activations f = ReLU(W_enc z + b_enc)."""
        return F.relu(self.encoder(z))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Linear decoder back to latent space."""
        return self.decoder(f)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(z)
        z_hat = self.decode(f)
        return z_hat, f


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class SAETrainConfig:
    d_hidden: int = 64
    l1_coeff: float = 3e-3
    lr: float = 1e-3
    batch_size: int = 4096
    num_epochs: int = 200
    device: str = "auto"


def train_sae(
    z: torch.Tensor,
    cfg: SAETrainConfig | None = None,
) -> tuple[SparseAutoencoder, dict]:
    """Train a SparseAutoencoder on the stack of latent vectors z (N, d_latent).

    Returns the trained SAE and a dict with the per-epoch loss trace,
    final reconstruction R^2, and final average L0 (feature sparsity) on
    the training set.
    """
    cfg = cfg or SAETrainConfig()
    torch.manual_seed(SEED)

    # Standardize so the L1 penalty is scale-invariant to the VAE's latent scale.
    z_mean = z.mean(dim=0, keepdim=True)
    z_std = z.std(dim=0, keepdim=True) + 1e-6
    z_norm = (z - z_mean) / z_std

    device = _pick_device(cfg.device)
    sae = SparseAutoencoder(d_latent=z.shape[1], d_hidden=cfg.d_hidden).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    dataset = TensorDataset(z_norm)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    loss_trace: list[float] = []
    for epoch in range(cfg.num_epochs):
        total = 0.0
        sae.train()
        for (batch,) in loader:
            batch = batch.to(device)
            z_hat, f = sae(batch)
            recon_loss = F.mse_loss(z_hat, batch)
            l1_loss = f.abs().mean()
            loss = recon_loss + cfg.l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.shape[0]
        loss_trace.append(total / len(dataset))

    # Final diagnostics on full data
    sae.eval()
    with torch.no_grad():
        z_norm_device = z_norm.to(device)
        z_hat, f = sae(z_norm_device)
        recon_r2 = _r2(z_norm_device, z_hat)
        mean_l0 = float((f > 0).float().mean().item() * cfg.d_hidden)

    diagnostics = {
        "final_loss": loss_trace[-1],
        "loss_trace": loss_trace,
        "recon_r2": recon_r2,
        "mean_l0": mean_l0,
        "d_hidden": cfg.d_hidden,
        "z_mean": z_mean,
        "z_std": z_std,
    }
    return sae.cpu(), diagnostics


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Feature interpretation: score each SAE feature against our 4 hypotheses
# ---------------------------------------------------------------------------


CARBONATE_KEYWORDS = (
    "nannofossil ooze", "foraminifera ooze", "carbonate ooze",
    "calcareous ooze", "nannofossil chalk", "foraminifera chalk",
    "chalk", "foraminifera limestone",
)
CLAY_KEYWORDS = (
    "clay", "silty clay", "claystone", "silty claystone",
    "mud", "mudstone", "silty mud", "silty mudstone",
)
VOLCANIC_KEYWORDS = (
    "ash", "tuff", "tephra", "volcanic", "lapilli", "volcaniclastic",
)


def _fraction_with_keyword(
    labels: np.ndarray,
    idx_encoder_classes: list[str],
    keywords: tuple[str, ...],
) -> np.ndarray:
    """For each sample, return 1.0 if its lithology name contains any keyword, else 0.0.

    Matching is done as substring membership on the decoded class name.
    """
    class_is_match = np.zeros(len(idx_encoder_classes), dtype=bool)
    for i, cname in enumerate(idx_encoder_classes):
        lc = cname.lower()
        if any(kw in lc for kw in keywords):
            class_is_match[i] = True
    return class_is_match[labels].astype(np.float32)


@dataclass
class FeatureScore:
    feature_idx: int
    depth_corr: float           # Pearson correlation with depth
    ngr_corr: float             # Pearson correlation with NGR (organic-rich proxy)
    carbonate_frac: float       # fraction of top-k activations that are carbonate
    clay_frac: float            # fraction of top-k activations that are clay
    volcanic_frac: float        # fraction of top-k activations that are volcanic
    carbonate_vs_clay: float    # carbonate_frac - clay_frac (signed, sensitive to direction)
    activation_frequency: float # fraction of samples where feature is nonzero
    mean_activation: float      # mean activation when nonzero


def score_features_against_targets(
    sae: SparseAutoencoder,
    z: torch.Tensor,                     # (N, d_latent) raw VAE latents
    labels: np.ndarray,                  # (N,) lithology class indices
    depth: np.ndarray,                   # (N,) depth_bins
    ngr: np.ndarray,                     # (N,) NGR feature values
    label_classes: list[str],            # list from label_encoder.classes_
    z_mean: torch.Tensor,                # standardization mean from training
    z_std: torch.Tensor,                 # standardization std from training
    top_k: int = 500,
) -> list[FeatureScore]:
    """Compute a FeatureScore for every SAE feature."""
    z_norm = (z - z_mean) / z_std
    with torch.no_grad():
        features = sae.encode(z_norm).numpy()  # (N, d_hidden)

    n, d_hidden = features.shape

    # Precompute lithology-class masks per sample (faster than reindexing inside loop).
    carbonate_mask = _fraction_with_keyword(labels, label_classes, CARBONATE_KEYWORDS)
    clay_mask = _fraction_with_keyword(labels, label_classes, CLAY_KEYWORDS)
    volcanic_mask = _fraction_with_keyword(labels, label_classes, VOLCANIC_KEYWORDS)

    depth_z = (depth - depth.mean()) / (depth.std() + 1e-9)
    ngr_z = (ngr - ngr.mean()) / (ngr.std() + 1e-9)

    scores: list[FeatureScore] = []
    for k in range(d_hidden):
        act = features[:, k]
        nonzero = act > 0

        # Activation statistics
        activation_frequency = float(nonzero.mean())
        mean_activation = float(act[nonzero].mean()) if nonzero.any() else 0.0

        # Pearson correlation with depth and NGR (standardized act vs standardized target)
        if act.std() > 1e-9:
            act_z = (act - act.mean()) / act.std()
            depth_corr = float((act_z * depth_z).mean())
            ngr_corr = float((act_z * ngr_z).mean())
        else:
            depth_corr = 0.0
            ngr_corr = 0.0

        # Lithology fractions at the top-k activating samples
        if top_k > 0 and top_k < n:
            top_idx = np.argpartition(act, -top_k)[-top_k:]
        else:
            top_idx = np.arange(n)
        carbonate_frac = float(carbonate_mask[top_idx].mean())
        clay_frac = float(clay_mask[top_idx].mean())
        volcanic_frac = float(volcanic_mask[top_idx].mean())

        scores.append(
            FeatureScore(
                feature_idx=k,
                depth_corr=depth_corr,
                ngr_corr=ngr_corr,
                carbonate_frac=carbonate_frac,
                clay_frac=clay_frac,
                volcanic_frac=volcanic_frac,
                carbonate_vs_clay=carbonate_frac - clay_frac,
                activation_frequency=activation_frequency,
                mean_activation=mean_activation,
            )
        )
    return scores


def find_best_matches(scores: list[FeatureScore]) -> dict[str, FeatureScore]:
    """Pick the single best feature for each of the 4 target hypotheses.

    Returns a dict with keys: 'depth', 'carbonate_vs_clay', 'volcanic', 'organic_rich'.
    The 'depth' entry uses absolute correlation (the direction of depth
    encoding doesn't matter), the other three use signed magnitudes
    because we care about features that activate *for* the target class.
    """
    if not scores:
        return {}

    best_depth = max(scores, key=lambda s: abs(s.depth_corr))
    best_carbonate_vs_clay = max(scores, key=lambda s: s.carbonate_vs_clay)
    best_volcanic = max(scores, key=lambda s: s.volcanic_frac)
    # Organic-rich proxy: high NGR correlation. Could also add siliceous-ooze
    # bias in the future.
    best_organic = max(scores, key=lambda s: s.ngr_corr)

    return {
        "depth": best_depth,
        "carbonate_vs_clay": best_carbonate_vs_clay,
        "volcanic": best_volcanic,
        "organic_rich": best_organic,
    }


# ---------------------------------------------------------------------------
# Steering: amplify a single SAE feature and return the modified latent
# ---------------------------------------------------------------------------


def steer(
    sae: SparseAutoencoder,
    z: torch.Tensor,                     # (N, d_latent) raw VAE latents
    feature_idx: int,
    magnitude: float,                    # additive boost applied to feature_idx
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
) -> torch.Tensor:
    """Return a modified latent z' where the specified SAE feature has been
    amplified by `magnitude` (in feature-space units), then decoded back
    through the SAE.

    The returned latent is in the ORIGINAL (un-standardized) latent space,
    ready to be passed into the VAE decoder for reconstruction.
    """
    z_norm = (z - z_mean) / z_std
    with torch.no_grad():
        f = sae.encode(z_norm)
        f_steered = f.clone()
        f_steered[:, feature_idx] = f_steered[:, feature_idx] + magnitude
        z_hat_norm = sae.decode(f_steered)
    z_hat = z_hat_norm * z_std + z_mean
    return z_hat
