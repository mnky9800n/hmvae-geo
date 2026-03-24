"""Loss functions and annealing schedules for VAE training."""

import torch


def heteroscedastic_reconstruction_loss(
    x: torch.Tensor, x_mu: torch.Tensor, x_logvar: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood under a heteroscedastic Gaussian decoder.

    -0.5 * sum_i [log_var_i + (x_i - mu_i)^2 / exp(log_var_i)]
    Averaged over batch.
    """
    return 0.5 * torch.mean(
        x_logvar + (x - x_mu).pow(2) / torch.exp(x_logvar)
    )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,1)), averaged over batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for lithology classification."""
    return torch.nn.functional.cross_entropy(logits, labels)


def beta_schedule(epoch: int, beta_start: float, beta_end: float, anneal_epochs: int) -> float:
    """Linear annealing from beta_start to beta_end over anneal_epochs."""
    if epoch >= anneal_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)


def alpha_schedule(epoch: int, alpha_start: float, alpha_end: float, anneal_epochs: int) -> float:
    """Linear annealing from alpha_start to alpha_end over anneal_epochs."""
    if epoch >= anneal_epochs:
        return alpha_end
    return alpha_start + (alpha_end - alpha_start) * (epoch / anneal_epochs)
