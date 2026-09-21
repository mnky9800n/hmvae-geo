"""Are the decoder's predicted variances calibrated uncertainties?

Reviewers asked two related questions: whether higher predicted variance
corresponds to larger reconstruction error, and what justifies clamping
the predicted log-variance to [-6, 6].

We never claimed calibrated uncertainty. The heteroscedastic decoder is
a training-time mechanism: the six inputs have very different noise
scales, and a single fixed variance lets whichever variable has the
largest residuals dictate the geometry of the latent space. Whether the
learned variances are *also* usable as uncertainties is a separate
question, and this module answers it rather than assuming it.

The answer on the shipped checkpoint is that they are not. Coverage of
nominal 68% and 95% intervals is far too high, the mean standardised
residual is far below one, and a homoscedastic baseline using the same
predicted means achieves marginally better negative log-likelihood. The
cause is visible in the clamp: reconstruction is near-perfect, residuals
are tiny, and the model would prefer variances smaller than the floor
permits, so most samples sit pinned at it.

That diagnosis makes a testable prediction, which `clamp_sweep` checks:
widening or removing the floor should release the saturation and improve
calibration. Reporting the result either way is the point.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr

from lilyvae.config import (
    LOGVAR_CLAMP_MAX,
    LOGVAR_CLAMP_MIN,
    TRAINING_VARIABLES,
)

LOG2PI = math.log(2.0 * math.pi)

#: Nominal central-interval levels to check coverage at.
NOMINAL_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.90, 0.95, 0.99)

#: Standard normal quantiles for those levels, z such that P(|Z| < z) = level.
_Z = {
    0.50: 0.6744897501960817,
    0.68: 0.9944578832097535,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def gaussian_nll(x: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """Elementwise Gaussian negative log-likelihood in nats."""
    return 0.5 * (LOG2PI + logvar + (x - mu) ** 2 / np.exp(logvar))


def evaluate_calibration(
    x: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
    clamp: tuple[float, float] = (LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX),
    variable_names: list[str] | None = None,
) -> dict:
    """Score predicted variances as uncertainties.

    Args:
        x: True standardised inputs, shape (n, d).
        mu: Predicted means, same shape.
        logvar: Predicted log-variances, same shape.
        clamp: The bounds the decoder applied, used to measure saturation.
        variable_names: Optional per-column names for the breakdown.

    Returns:
        dict with negative log-likelihood against a homoscedastic
        baseline using the same means, empirical coverage at each nominal
        level, the mean squared standardised residual (1.0 when
        calibrated), the fraction of predictions pinned at each clamp
        bound, and the rank correlation between predicted variance and
        absolute error.
    """
    names = variable_names or list(TRAINING_VARIABLES)
    sigma = np.exp(0.5 * logvar)
    resid = x - mu
    standardised = resid / sigma

    # A homoscedastic baseline with one maximum-likelihood variance per
    # variable, reusing the same predicted means. If the learned
    # per-sample variances carry signal, they should beat it.
    # Floored: a variable reconstructed exactly gives a zero maximum-
    # likelihood variance, and the resulting 0/0 would poison the mean.
    sigma2_homo = np.maximum((resid ** 2).mean(axis=0, keepdims=True), 1e-12)
    nll_hetero = gaussian_nll(x, mu, logvar).mean()
    nll_homo = gaussian_nll(
        x, mu, np.log(sigma2_homo) * np.ones_like(logvar)
    ).mean()

    coverage = {}
    for level in NOMINAL_LEVELS:
        z = _Z[level]
        empirical = float(np.mean(np.abs(standardised) < z))
        coverage[f"{level:.2f}"] = {
            "nominal": level,
            "empirical": empirical,
            "excess": empirical - level,
        }

    lo, hi = clamp
    at_floor = float(np.mean(logvar <= lo + 1e-6))
    at_ceiling = float(np.mean(logvar >= hi - 1e-6))

    # Does a larger predicted variance mean a larger error? Rank
    # correlation, because only the ordering matters for this question.
    flat_var, flat_err = logvar.ravel(), np.abs(resid).ravel()
    if flat_var.std() < 1e-12:
        rho = float("nan")  # fully saturated, no ordering to correlate
    else:
        rho = float(spearmanr(flat_var, flat_err).statistic)

    per_variable = {}
    for i, name in enumerate(names[: x.shape[1]]):
        col_var = logvar[:, i]
        per_variable[name] = {
            "mean_standardised_residual_sq": float((standardised[:, i] ** 2).mean()),
            "fraction_at_clamp_floor": float(np.mean(col_var <= lo + 1e-6)),
            "nll_heteroscedastic": float(gaussian_nll(
                x[:, i], mu[:, i], logvar[:, i]
            ).mean()),
        }

    return {
        "n_samples": int(x.shape[0]),
        "clamp": [float(lo), float(hi)],
        "nll_heteroscedastic": float(nll_hetero),
        "nll_homoscedastic_same_means": float(nll_homo),
        # Positive means the learned per-sample variances helped.
        "nll_gain_from_learned_variance": float(nll_homo - nll_hetero),
        "coverage": coverage,
        # 1.0 when calibrated; below 1 means the intervals are too wide.
        "mean_standardised_residual_sq": float((standardised ** 2).mean()),
        "fraction_at_clamp_floor": at_floor,
        "fraction_at_clamp_ceiling": at_ceiling,
        "spearman_variance_vs_abs_error": rho,
        "per_variable": per_variable,
    }


@torch.no_grad()
def calibration_from_model(vae, features: torch.Tensor) -> dict:
    """Run `evaluate_calibration` on a trained VAE and its training inputs.

    Decodes from the posterior mean rather than from a sample. `forward`
    reparameterises, so decoding through it makes every call return
    slightly different coverage and NLL figures, and a calibration number
    that moves when nothing changed is not a number worth reporting.
    Downstream evaluation already uses the deterministic `get_embeddings`
    path, so this also matches how the embeddings are produced.
    """
    vae.eval()
    z_mu, _ = vae.encoder(features)
    mu, logvar = vae.decoder(z_mu)
    clamp = getattr(vae.decoder, "logvar_clamp", (LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX))
    return evaluate_calibration(
        features.numpy().astype(np.float64),
        mu.numpy().astype(np.float64),
        logvar.numpy().astype(np.float64),
        clamp=tuple(clamp),
    )


def clamp_sweep(
    features: torch.Tensor,
    labels: torch.Tensor,
    clamps: tuple[tuple[float, float], ...] = (
        (-6.0, 6.0),
        (-12.0, 6.0),
        (-20.0, 20.0),
    ),
    num_epochs: int | None = None,
    device: str = "auto",
) -> dict:
    """Retrain with different log-variance bounds and re-check calibration.

    The diagnosis is that the floor binds because reconstruction is
    near-perfect, so the model wants smaller variances than it is
    allowed. If that is right, widening the floor should reduce the
    fraction pinned at it and move coverage towards nominal. If coverage
    does not improve, the variances are uninformative for a reason the
    clamp does not explain, and the paper should say so.
    """
    from lilyvae.config import (
        LEARNING_RATE,
        MASK_FRACTION,
        MAX_GRAD_NORM,
        NUM_EPOCHS,
        WEIGHT_DECAY,
    )
    from lilyvae.evaluation.ablations import _train_ablation_model
    from lilyvae.evaluation.bootstrap import _make_vae_dataloader
    from lilyvae.model.vae import LilyVAE
    from lilyvae.training.masking import LearnableMasking
    from lilyvae.training.trainer import get_device

    dev = get_device(device)
    epochs = NUM_EPOCHS if num_epochs is None else num_epochs
    results = {}

    for clamp in clamps:
        key = f"[{clamp[0]:g},{clamp[1]:g}]"
        print(f"\n  clamp {key}: training {epochs} epochs")
        torch.manual_seed(42)
        vae = LilyVAE(logvar_clamp=clamp)
        masking = LearnableMasking(features.shape[1], MASK_FRACTION)
        vae = _train_ablation_model(
            dataloader=_make_vae_dataloader(features, labels),
            vae=vae,
            masking=masking,
            decoder_type="heteroscedastic",
            beta_schedule_type="annealing",
            alpha_weight=0.0,
            num_epochs=epochs,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            device=dev,
        )
        vae = vae.cpu()
        res = calibration_from_model(vae, features)
        results[key] = res
        print(
            f"    at floor {res['fraction_at_clamp_floor']:.3f}  "
            f"68% coverage {res['coverage']['0.68']['empirical']:.3f}  "
            f"mean std resid^2 {res['mean_standardised_residual_sq']:.3f}  "
            f"NLL {res['nll_heteroscedastic']:.3f}"
        )

    return results
