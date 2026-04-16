"""Non-VAE baseline models for comparison against LilyVAE.

Three baselines are implemented:

PCA
    Fits PCA on the 6 training features and uses the top `latent_dim`
    principal components as embeddings. Reconstruction uses
    inverse_transform. No neural network training required.

TabNetBaseline
    TabNet (Arik & Pfister, 2021) with unsupervised mask-reconstruct
    pretraining. No labels required. Embeddings are the summed decision
    step outputs from the TabNet encoder.

SCARFBaseline
    SCARF (Bahri et al., ICML 2022) contrastive self-supervised learning.
    No labels required. Corrupts random features with marginal samples and
    learns embeddings via NT-Xent contrastive loss. Does not reconstruct
    inputs, so reconstruction R^2 is not applicable.

All implement get_embeddings(x) -> Tensor. Reconstruction-based models also
return a dict with key "x_mu" from forward/call, matching the interface that
evaluate_reconstruction expects.
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
    TRAINING_VARIABLES,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    BATCH_SIZE,
    SEED,
)
from lilyvae.training.trainer import get_device
from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot
from lilyvae.data.dataset import get_dataloader
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
        n_components = min(self.latent_dim, X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=SEED)
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
# TabNet unsupervised baseline
# ---------------------------------------------------------------------------


class TabNetBaseline:
    """TabNet with unsupervised mask-reconstruct pretraining (no labels).

    Uses pytorch_tabnet.pretraining.TabNetPretrainer. Embeddings are the
    summed decision-step outputs from the TabNet encoder, giving a vector
    of dimensionality n_d per sample.

    Note: TabNet's pretraining objective is masked-feature reconstruction
    (predict masked values from unmasked ones), not full autoencoder
    reconstruction. The predict() output collapses toward feature means
    for unmasked positions, so reconstruction R^2 is not a meaningful
    metric for this baseline and is reported as NaN.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_steps: int = 3,
        pretraining_ratio: float = 0.5,
    ):
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.pretraining_ratio = pretraining_ratio
        self.model = None

    def fit(self, dataloader: DataLoader, max_epochs: int = NUM_EPOCHS) -> None:
        from pytorch_tabnet.pretraining import TabNetPretrainer

        all_features = []
        for features, _, _ in dataloader:
            all_features.append(features.numpy())
        X = np.concatenate(all_features, axis=0)

        self.model = TabNetPretrainer(
            n_d=self.latent_dim,
            n_a=self.latent_dim,
            n_steps=self.n_steps,
            momentum=0.3,
            mask_type="entmax",
            verbose=0,
            seed=SEED,
        )
        self.model.fit(
            X,
            max_epochs=max_epochs,
            batch_size=1024,
            virtual_batch_size=128,
            pretraining_ratio=self.pretraining_ratio,
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # TabNetPretrainer places its network on whatever device its internal
        # `device_name="auto"` picked (typically cuda:0 when CUDA is available).
        # The caller passes x on CPU, so we mirror the network's device before
        # the forward pass and return embeddings on CPU for downstream eval
        # which uses sklearn/numpy.
        net = self.model.network
        net.eval()
        net_device = next(net.parameters()).device
        with torch.no_grad():
            x = x.to(net_device)
            steps_output, _ = net.encoder(x)
            emb = sum(steps_output)
        return emb.cpu()

    def eval(self) -> "TabNetBaseline":
        if self.model is not None:
            self.model.network.eval()
        return self


# ---------------------------------------------------------------------------
# SCARF contrastive baseline
# ---------------------------------------------------------------------------


class _SCARFEncoder(nn.Module):
    """MLP encoder for SCARF."""

    def __init__(self, input_dim: int, hidden_dim: int, emb_dim: int, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SCARFProjector(nn.Module):
    """Projection head for SCARF (discarded after training)."""

    def __init__(self, emb_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _nt_xent_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR-style)."""
    batch_size = z_anchor.shape[0]
    z = torch.cat([z_anchor, z_positive], dim=0)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    # Mask out self-similarity
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, -1e9)
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size),
    ]).to(z.device)
    return F.cross_entropy(sim, labels)


class SCARFBaseline(nn.Module):
    """SCARF: contrastive self-supervised learning for tabular data (no labels).

    Corrupts random features with values drawn from per-feature marginal
    distributions and trains with NT-Xent contrastive loss. The projection
    head is discarded after training; the encoder output is the embedding.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 128,
        encoder_depth: int = 3,
        corruption_rate: float = 0.6,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.encoder = _SCARFEncoder(input_dim, hidden_dim, latent_dim, encoder_depth)
        self.projector = _SCARFProjector(latent_dim)
        self.corruption_rate = corruption_rate
        self.temperature = temperature
        self.marginals: torch.Tensor | None = None

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        mask = torch.rand(B, D, device=x.device) < self.corruption_rate
        indices = torch.randint(0, len(self.marginals), (B, D))
        marginal_samples = torch.stack(
            [self.marginals[indices[:, d], d] for d in range(D)], dim=1
        ).to(x.device)
        return torch.where(mask, marginal_samples, x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_corrupted = self._corrupt(x)
        z_anchor = self.projector(self.encoder(x))
        z_positive = self.projector(self.encoder(x_corrupted))
        return z_anchor, z_positive

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Neural baseline training
# ---------------------------------------------------------------------------



def _train_scarf(
    model: SCARFBaseline,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> SCARFBaseline:
    """Train a SCARFBaseline with NT-Xent contrastive loss."""
    # Store marginal distributions from all training data
    all_features = []
    for features, _, _ in dataloader:
        all_features.append(features)
    all_X = torch.cat(all_features, dim=0)
    model.marginals = all_X

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for epoch in range(num_epochs):
        model.train()
        for features, _, _ in dataloader:
            features = features.to(device)
            z_anchor, z_positive = model(features)
            loss = _nt_xent_loss(z_anchor, z_positive, model.temperature)

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
    metrics = baseline.evaluate(dataloader)

    # Zero-shot evaluation
    from lilyvae.config import PROCESSED_DIR
    import os
    zs_path = PROCESSED_DIR / "zeroshot_targets.pt"
    if zs_path.exists():
        all_z, _ = _extract_all(baseline, dataloader)
        zeroshot_targets = torch.load(zs_path, weights_only=False)
        print("  Zero-shot prediction:")
        zs_results = evaluate_all_zeroshot(all_z, zeroshot_targets)
        zs_r2s = [v["r2"] for v in zs_results.values() if not np.isnan(v["r2"])]
        metrics["zeroshot_r2_avg"] = float(np.mean(zs_r2s)) if zs_r2s else float("nan")
        metrics["zeroshot_n_variables"] = len(zs_r2s)

    return metrics



def evaluate_tabnet_baseline(
    latent_dim: int = LATENT_DIM,
    dataloader: DataLoader | None = None,
    num_epochs: int = NUM_EPOCHS,
) -> dict:
    """Train TabNet unsupervised pretrainer and compute evaluation metrics.

    TabNet uses mask-reconstruct pretraining (no labels required).
    Reconstruction R^2 is not reported because TabNet's objective is
    masked-feature prediction, not full autoencoder reconstruction.

    Args:
        latent_dim: Embedding dimensionality (TabNet n_d parameter).
        dataloader: Optional pre-built DataLoader.
        num_epochs: Training epochs.

    Returns:
        Metrics dict with model name, ari, auc_macro, and zero-shot R^2.
    """
    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[baseline] TabNet (unsupervised)  latent_dim={latent_dim}")
    baseline = TabNetBaseline(latent_dim)
    baseline.fit(dataloader, max_epochs=num_epochs)

    # Embeddings for clustering/classification/zeroshot
    all_z, all_labels = _extract_all(baseline, dataloader)
    clust_metrics = evaluate_clustering(all_z, all_labels)
    cls_metrics = evaluate_classification(all_z, all_labels)

    metrics = {
        "model": "tabnet_unsupervised",
        "latent_dim": latent_dim,
        "r2_avg": float("nan"),  # TabNet does masked prediction, not full reconstruction
        "ari": clust_metrics["ari"],
        "auc_macro": cls_metrics["auc_macro"],
    }

    from lilyvae.config import PROCESSED_DIR
    zs_path = PROCESSED_DIR / "zeroshot_targets.pt"
    if zs_path.exists():
        zeroshot_targets = torch.load(zs_path, weights_only=False)
        print("  Zero-shot prediction:")
        zs_results = evaluate_all_zeroshot(all_z, zeroshot_targets)
        zs_r2s = [v["r2"] for v in zs_results.values() if not np.isnan(v["r2"])]
        metrics["zeroshot_r2_avg"] = float(np.mean(zs_r2s)) if zs_r2s else float("nan")
        metrics["zeroshot_n_variables"] = len(zs_r2s)

    return metrics


def evaluate_scarf_baseline(
    latent_dim: int = LATENT_DIM,
    dataloader: DataLoader | None = None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    device: str = "auto",
) -> dict:
    """Train SCARF contrastive baseline and compute evaluation metrics.

    SCARF uses contrastive self-supervised learning (no labels required).
    Reconstruction R^2 is not applicable since SCARF does not reconstruct.

    Args:
        latent_dim: Embedding dimensionality.
        dataloader: Optional pre-built DataLoader.
        num_epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Gradient clipping threshold.
        device: Compute device string.

    Returns:
        Metrics dict with model name, ari, auc_macro, and zero-shot R^2.
    """
    torch.manual_seed(SEED)
    dev = get_device(device)

    if dataloader is None:
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[baseline] SCARF (contrastive)  latent_dim={latent_dim}")
    model = SCARFBaseline(INPUT_DIM, latent_dim)
    model = _train_scarf(
        model, dataloader, num_epochs, lr, weight_decay, max_grad_norm, dev
    )

    all_z, all_labels = _extract_all_nn(model, dataloader, dev)
    clust_metrics = evaluate_clustering(all_z, all_labels)
    cls_metrics = evaluate_classification(all_z, all_labels)

    metrics = {
        "model": "scarf_contrastive",
        "latent_dim": latent_dim,
        "r2_avg": float("nan"),  # SCARF does not reconstruct
        "ari": clust_metrics["ari"],
        "auc_macro": cls_metrics["auc_macro"],
    }

    from lilyvae.config import PROCESSED_DIR
    zs_path = PROCESSED_DIR / "zeroshot_targets.pt"
    if zs_path.exists():
        zeroshot_targets = torch.load(zs_path, weights_only=False)
        print("  Zero-shot prediction:")
        zs_results = evaluate_all_zeroshot(all_z, zeroshot_targets)
        zs_r2s = [v["r2"] for v in zs_results.values() if not np.isnan(v["r2"])]
        metrics["zeroshot_r2_avg"] = float(np.mean(zs_r2s)) if zs_r2s else float("nan")
        metrics["zeroshot_n_variables"] = len(zs_r2s)

    return metrics


def run_all_baselines(
    latent_dim: int = LATENT_DIM,
    num_epochs: int = NUM_EPOCHS,
    device: str = "auto",
) -> list[dict]:
    """Evaluate all three baselines and return a list of result dicts.

    Returns:
        List of metrics dicts: [pca, tabnet_unsupervised, scarf_contrastive].
    """
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    results = [
        evaluate_pca_baseline(latent_dim=latent_dim, dataloader=dataloader),
        evaluate_tabnet_baseline(
            latent_dim=latent_dim,
            dataloader=dataloader,
            num_epochs=num_epochs,
        ),
        evaluate_scarf_baseline(
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
