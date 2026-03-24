"""Training loop for LilyVAE."""

import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lilyvae.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    BETA_START,
    BETA_END,
    BETA_ANNEAL_EPOCHS,
    ALPHA_START,
    ALPHA_END,
    ALPHA_ANNEAL_EPOCHS,
    MASK_FRACTION,
    LATENT_DIM,
    HIDDEN_DIMS,
    INPUT_DIM,
    NUM_LITHOLOGY_CLASSES,
    CHECKPOINT_DIR,
    SEED,
)
from lilyvae.model.vae import LilyVAE
from lilyvae.model.heads import LithologyClassifier
from lilyvae.training.loss import (
    heteroscedastic_reconstruction_loss,
    kl_divergence,
    classification_loss,
    beta_schedule,
    alpha_schedule,
)
from lilyvae.training.masking import LearnableMasking


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def train_model(
    dataloader: DataLoader,
    model_type: str = "unsupervised",
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    mask_fraction: float = MASK_FRACTION,
    latent_dim: int = LATENT_DIM,
    hidden_dims: list[int] | None = None,
    device: str = "auto",
) -> dict:
    """Train a LilyVAE model.

    Args:
        dataloader: DataLoader yielding (features, labels, indices)
        model_type: "unsupervised" or "semi-supervised"
        num_epochs: Total training epochs
        lr: Learning rate
        weight_decay: Adam weight decay
        max_grad_norm: Gradient clipping max norm
        mask_fraction: Fraction of input features to mask
        latent_dim: Latent space dimensionality
        hidden_dims: Encoder/decoder hidden layer sizes
        device: "auto", "cuda", "mps", or "cpu"

    Returns:
        Dict with trained model, masking module, classifier (if semi-supervised),
        and training history.
    """
    torch.manual_seed(SEED)
    dev = get_device(device)
    hidden_dims = hidden_dims or list(HIDDEN_DIMS)

    vae = LilyVAE(INPUT_DIM, latent_dim, hidden_dims).to(dev)
    masking = LearnableMasking(INPUT_DIM, mask_fraction).to(dev)

    params = list(vae.parameters()) + list(masking.parameters())
    classifier = None
    if model_type == "semi-supervised":
        classifier = LithologyClassifier(latent_dim, NUM_LITHOLOGY_CLASSES).to(dev)
        params += list(classifier.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "loss": [], "recon": [], "kl": [], "beta": []}
    if classifier:
        history["cls"] = []
        history["alpha"] = []

    for epoch in range(num_epochs):
        vae.train()
        masking.train()
        if classifier:
            classifier.train()

        beta = beta_schedule(epoch, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)
        alpha = alpha_schedule(epoch, ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS) if classifier else 0.0

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_cls = 0.0
        n_batches = 0

        for features, labels, _ in dataloader:
            features = features.to(dev)
            labels = labels.to(dev)

            masked_features = masking(features)
            out = vae(masked_features)

            recon = heteroscedastic_reconstruction_loss(features, out["x_mu"], out["x_logvar"])
            kl = kl_divergence(out["z_mu"], out["z_logvar"])
            loss = recon + beta * kl

            if classifier:
                logits = classifier(out["z"])
                cls = classification_loss(logits, labels)
                loss = loss + alpha * cls
                epoch_cls += cls.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss / n_batches)
        history["recon"].append(epoch_recon / n_batches)
        history["kl"].append(epoch_kl / n_batches)
        history["beta"].append(beta)
        if classifier:
            history["cls"].append(epoch_cls / n_batches)
            history["alpha"].append(alpha)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1:3d}/{num_epochs} | loss={epoch_loss/n_batches:.4f} recon={epoch_recon/n_batches:.4f} kl={epoch_kl/n_batches:.4f} beta={beta:.6f}"
            if classifier:
                msg += f" cls={epoch_cls/n_batches:.4f} alpha={alpha:.6f}"
            print(msg)

    return {
        "vae": vae,
        "masking": masking,
        "classifier": classifier,
        "history": history,
    }


def save_checkpoint(result: dict, model_type: str) -> None:
    """Save model checkpoint and training history."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = model_type.replace("-", "_")

    torch.save(result["vae"].state_dict(), CHECKPOINT_DIR / f"{prefix}_vae.pt")
    torch.save(result["masking"].state_dict(), CHECKPOINT_DIR / f"{prefix}_masking.pt")
    if result["classifier"] is not None:
        torch.save(result["classifier"].state_dict(), CHECKPOINT_DIR / f"{prefix}_classifier.pt")

    with open(CHECKPOINT_DIR / f"{prefix}_history.json", "w") as f:
        json.dump(result["history"], f)


def extract_embeddings(vae: LilyVAE, dataloader: DataLoader, device: str = "auto") -> torch.Tensor:
    """Extract deterministic embeddings (mu) for the full dataset."""
    dev = get_device(device)
    vae.eval()
    all_embeddings = []
    with torch.no_grad():
        for features, _, _ in dataloader:
            features = features.to(dev)
            z_mu = vae.get_embeddings(features)
            all_embeddings.append(z_mu.cpu())
    return torch.cat(all_embeddings, dim=0)
