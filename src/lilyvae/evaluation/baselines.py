"""Non-VAE baseline models for comparison against LilyVAE.

Three baselines are implemented:

PCA
    Fits PCA on the 6 training features and uses the top `latent_dim`
    principal components as embeddings. Reconstruction uses
    inverse_transform. No neural network training required.

LinearAutoencoder
    Same hidden-layer widths as LilyVAE but with no activation functions
    and no KL term. Trained with MSE loss only. The bottleneck IS the
    embedding — this is a classical linear autoencoder and collapses to a
    subspace equivalent to PCA given sufficient training, making it a useful
    sanity-check for the nonlinear models.

DeterministicAutoencoder
    Same architecture as LilyVAE (with ReLU activations) but no
    reparameterization trick and no KL term. The encoder directly outputs
    the embedding rather than a distribution. Trained with MSE loss only.
    Isolates the effect of the variational objective.

All three implement get_embeddings(x) -> Tensor and a forward method that
returns a dict with key "x_mu", matching the interface that
evaluate_reconstruction expects (duck typing — no shared base class needed).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from lilyvae.config import (
    INPUT_DIM,
    LATENT_DIM,
    HIDDEN_DIMS,
    TRAINING_VARIABLES,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    BATCH_SIZE,
    SEED,
)
from lilyvae.training.trainer import get_device
from lilyvae.data.dataset import get_dataloader
from lilyvae.evaluation.reconstruction import evaluate_reconstruction
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.classification import evaluate_classification


# ---------------------------------------------------------------------------
# PCA baseline
# ---------------------------------------------------------------------------


class PCABaseline:
    """Scikit-learn PCA wrapped in the project's evaluation interface.

    Attributes:
        pca: The fitted sklearn PCA object.
        latent_dim: Number of principal components retained.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        self.latent_dim = latent_dim
        self.pca: PCA | None = None

    # ------------------------------------------------------------------
    # Training (fit)
    # ------------------------------------------------------------------

    def fit(self, dataloader: DataLoader) -> None:
        """Fit PCA on all features in the dataloader."""
        all_features = []
        for features, _, _ in dataloader:
            all_features.append(features.numpy())
        X = np.concatenate(all_features, axis=0)
        self.pca = PCA(n_components=self.latent_dim, random_state=SEED)
        self.pca.fit(X)

    # ------------------------------------------------------------------
    # Evaluation interface
    # ------------------------------------------------------------------

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Project x into the PCA subspace."""
        if self.pca is None:
            raise RuntimeError("PCA has not been fitted. Call fit() first.")
        z = self.pca.transform(x.numpy())
        return torch.from_numpy(z.astype(np.float32))

    # evaluate_reconstruction calls vae(features) and reads out["x_mu"].
    # We implement __call__ to satisfy that duck-typed interface.
    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.pca is None:
            raise RuntimeError("PCA has not been fitted. Call fit() first.")
        z = self.pca.transform(x.numpy())
        x_recon = self.pca.inverse_transform(z)
        return {"x_mu": torch.from_numpy(x_recon.astype(np.float32))}

    def eval(self) -> "PCABaseline":
        """No-op: matches the nn.Module.eval() call in evaluate_reconstruction."""
        return self

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Compute R², ARI, and AUC for this PCA baseline."""
        self.fit(dataloader)

        # Reconstruction — evaluate_reconstruction iterates the dataloader
        # and calls vae(features). PCABaseline.__call__ handles that.
        recon_metrics = _evaluate_reconstruction_baseline(self, dataloader)

        all_z, all_labels = _extract_all(self, dataloader)
        clust_metrics = evaluate_clustering(all_z, all_labels)
        cls_metrics = evaluate_classification(all_z, all_labels)

        return {
            "model": "pca",
            "latent_dim": self.latent_dim,
            **recon_metrics,
            "ari": clust_metrics["ari"],
            "auc_macro": cls_metrics["auc_macro"],
        }


# ---------------------------------------------------------------------------
# Neural network baselines
# ---------------------------------------------------------------------------


class LinearAutoencoder(nn.Module):
    """Linear autoencoder: same widths as LilyVAE, no activations, no KL.

    The encoder maps input -> hidden layers (linear only) -> bottleneck.
    The decoder mirrors that path back to input dimension.
    Trained with MSE loss.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or list(HIDDEN_DIMS)

        # Encoder: input -> hidden layers -> bottleneck (linear only)
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder_net = nn.Sequential(*enc_layers)

        # Decoder: bottleneck -> hidden layers -> output (linear only)
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder_net(x)
        x_recon = self.decoder_net(z)
        return {"x_mu": x_recon, "z": z}

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)


class DeterministicAutoencoder(nn.Module):
    """Nonlinear deterministic autoencoder: ReLU activations, no KL term.

    The encoder directly outputs the embedding (no mu/logvar split, no
    reparameterization). This isolates the effect of the variational
    objective: comparing against LilyVAE tells us how much the KL
    regularizer contributes to downstream performance.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or list(HIDDEN_DIMS)

        # Encoder with ReLU activations
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            enc_layers.append(nn.ReLU())
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder_net = nn.Sequential(*enc_layers)

        # Decoder with ReLU activations
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            dec_layers.append(nn.ReLU())
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder_net(x)
        x_recon = self.decoder_net(z)
        return {"x_mu": x_recon, "z": z}

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)


# ---------------------------------------------------------------------------
# Neural baseline training
# ---------------------------------------------------------------------------


def _train_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> nn.Module:
    """Train a LinearAutoencoder or DeterministicAutoencoder with MSE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for epoch in range(num_epochs):
        model.train()
        for features, _, _ in dataloader:
            features = features.to(device)
            out = model(features)
            loss = F.mse_loss(out["x_mu"], features)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{num_epochs}")

    return model


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def evaluate_pca_baseline(
    latent_dim: int = LATENT_DIM,
    dataloader: DataLoader | None = None,
) -> dict:
    """Fit PCA and compute evaluation metrics.

    Args:
        latent_dim: Number of principal components to retain.
        dataloader: Optional pre-built DataLoader (shuffle=False recommended).

    Returns:
        Metrics dict with model name, r2_avg, ari, auc_macro, and per-variable R².
    """
    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[baseline] PCA  latent_dim={latent_dim}")
    baseline = PCABaseline(latent_dim)
    return baseline.evaluate(dataloader)


def evaluate_linear_autoencoder(
    latent_dim: int = LATENT_DIM,
    dataloader: DataLoader | None = None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    device: str = "auto",
) -> dict:
    """Train and evaluate a LinearAutoencoder baseline.

    Args:
        latent_dim: Bottleneck dimensionality.
        dataloader: Optional pre-built DataLoader (shuffle=False recommended).
        num_epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Gradient clipping threshold.
        device: "auto", "cuda", "mps", or "cpu".

    Returns:
        Metrics dict with model name, r2_avg, ari, auc_macro.
    """
    torch.manual_seed(SEED)
    dev = get_device(device)

    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[baseline] LinearAutoencoder  latent_dim={latent_dim}")
    model = LinearAutoencoder(INPUT_DIM, latent_dim)
    model = _train_autoencoder(
        model, dataloader, num_epochs, lr, weight_decay, max_grad_norm, dev
    )

    eval_loader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)
    recon_metrics = evaluate_reconstruction(model, eval_loader, dev)

    all_z, all_labels = _extract_all_nn(model, eval_loader, dev)
    clust_metrics = evaluate_clustering(all_z, all_labels)
    cls_metrics = evaluate_classification(all_z, all_labels)

    return {
        "model": "linear_autoencoder",
        "latent_dim": latent_dim,
        **recon_metrics,
        "ari": clust_metrics["ari"],
        "auc_macro": cls_metrics["auc_macro"],
    }


def evaluate_deterministic_autoencoder(
    latent_dim: int = LATENT_DIM,
    dataloader: DataLoader | None = None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    device: str = "auto",
) -> dict:
    """Train and evaluate a DeterministicAutoencoder baseline.

    Args:
        latent_dim: Bottleneck dimensionality.
        dataloader: Optional pre-built DataLoader (shuffle=False recommended).
        num_epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Gradient clipping threshold.
        device: "auto", "cuda", "mps", or "cpu".

    Returns:
        Metrics dict with model name, r2_avg, ari, auc_macro.
    """
    torch.manual_seed(SEED)
    dev = get_device(device)

    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[baseline] DeterministicAutoencoder  latent_dim={latent_dim}")
    model = DeterministicAutoencoder(INPUT_DIM, latent_dim)
    model = _train_autoencoder(
        model, dataloader, num_epochs, lr, weight_decay, max_grad_norm, dev
    )

    eval_loader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)
    recon_metrics = evaluate_reconstruction(model, eval_loader, dev)

    all_z, all_labels = _extract_all_nn(model, eval_loader, dev)
    clust_metrics = evaluate_clustering(all_z, all_labels)
    cls_metrics = evaluate_classification(all_z, all_labels)

    return {
        "model": "deterministic_autoencoder",
        "latent_dim": latent_dim,
        **recon_metrics,
        "ari": clust_metrics["ari"],
        "auc_macro": cls_metrics["auc_macro"],
    }


def run_all_baselines(
    latent_dim: int = LATENT_DIM,
    num_epochs: int = NUM_EPOCHS,
    device: str = "auto",
) -> list[dict]:
    """Evaluate all three baselines and return a list of result dicts.

    Args:
        latent_dim: Embedding dimensionality for all baselines.
        num_epochs: Training epochs for the neural baselines.
        device: Compute device string.

    Returns:
        List of metrics dicts, one per baseline, in order:
        [pca, linear_autoencoder, deterministic_autoencoder].
    """
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    results = [
        evaluate_pca_baseline(latent_dim=latent_dim, dataloader=dataloader),
        evaluate_linear_autoencoder(
            latent_dim=latent_dim,
            dataloader=dataloader,
            num_epochs=num_epochs,
            device=device,
        ),
        evaluate_deterministic_autoencoder(
            latent_dim=latent_dim,
            dataloader=dataloader,
            num_epochs=num_epochs,
            device=device,
        ),
    ]
    return results


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _evaluate_reconstruction_baseline(
    model: PCABaseline, dataloader: DataLoader
) -> dict[str, float]:
    """R² evaluation for PCA (sklearn model, not nn.Module).

    Reimplements the core loop from evaluate_reconstruction without the
    torch.no_grad() and device-placement machinery that only applies to
    nn.Module models.
    """
    all_true = []
    all_pred = []
    for features, _, _ in dataloader:
        out = model(features)
        all_true.append(features.numpy())
        all_pred.append(out["x_mu"].numpy())

    true = np.concatenate(all_true, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    var_names = (
        TRAINING_VARIABLES
        if len(TRAINING_VARIABLES) == true.shape[1]
        else [f"var_{i}" for i in range(true.shape[1])]
    )

    results: dict[str, float] = {}
    for i, name in enumerate(var_names):
        results[f"r2_{name}"] = float(r2_score(true[:, i], pred[:, i]))
    results["r2_avg"] = float(
        np.mean([v for k, v in results.items() if k.startswith("r2_")])
    )
    return results


def _extract_all(
    model: PCABaseline, dataloader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect embeddings and labels for the PCA baseline."""
    all_z = []
    all_labels = []
    for features, labels, _ in dataloader:
        z = model.get_embeddings(features)
        all_z.append(z)
        all_labels.append(labels)
    return torch.cat(all_z, dim=0), torch.cat(all_labels, dim=0)


def _extract_all_nn(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect embeddings and labels for an nn.Module autoencoder."""
    model.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features.to(device)
            z = model.get_embeddings(features)
            all_z.append(z.cpu())
            all_labels.append(labels)
    return torch.cat(all_z, dim=0), torch.cat(all_labels, dim=0)
