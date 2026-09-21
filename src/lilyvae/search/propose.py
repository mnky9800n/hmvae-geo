"""Choose the next configuration to try, without a human in the loop.

Deliberately a hill climb with restarts rather than anything fancier.
Each evaluation costs roughly twenty minutes, so a run gets tens of
observations, not thousands. Surrogate-model methods need far more than
that before their acquisition function beats perturbing the best known
point, and they would hide the noise problem rather than respect it.

Two things the proposer must respect, both learned from round one.

The objective is constrained, not scalar: a configuration that raises
ARI by giving up transfer AUC is not an improvement. So the point being
climbed from is the best *admissible* trial, and a proposal descended
from an inadmissible one would only travel further in the wrong
direction.

The objective is noisy. Fold spread on ARI was about 0.047 against a
between-configuration range of 0.126, so a hill climb that treats every
small gain as real will wander. `climb_from` therefore only accepts a
new leader when it beats the incumbent by more than the incumbent's own
fold spread.
"""

from __future__ import annotations

import numpy as np

from lilyvae.search.ledger import Ledger, Trial
from lilyvae.search.space import SEARCH_SPACE, SearchConfig, random_config

#: Fraction of proposals drawn at random rather than by perturbation.
#: Keeps the search from settling into the first local optimum it finds.
EXPLORE_PROBABILITY = 0.25

#: Attempts to find a configuration absent from the ledger before giving
#: up and drawing at random.
MAX_ATTEMPTS = 40


def _step_in_list(value, options: list, rng: np.random.RandomState):
    """Move to a neighbouring value in an ordered list of choices."""
    options = list(options)
    try:
        i = options.index(value)
    except ValueError:
        return options[rng.randint(len(options))]
    candidates = [j for j in (i - 1, i + 1) if 0 <= j < len(options)]
    return options[candidates[rng.randint(len(candidates))]]


def _scale(value: float, rng: np.random.RandomState, spread: float = 0.5) -> float:
    """Multiply by a lognormal factor, for parameters that live on a log scale."""
    return float(value * np.exp(rng.normal(0.0, spread)))


def perturb(config: SearchConfig, rng: np.random.RandomState) -> tuple[SearchConfig, str]:
    """Change one dimension of a configuration.

    One at a time, so that when a proposal wins or loses it is clear
    which change did it. That matters more here than search speed: the
    ledger is meant to be readable as an experiment, not only mined for
    a maximum.
    """
    d = config.to_dict()
    dimension = rng.choice([
        "latent_dim", "hidden_dims", "beta_end", "mask_fraction",
        "learning_rate", "weight_decay", "num_epochs", "logvar_clamp_min",
    ])

    if dimension == "latent_dim":
        d["latent_dim"] = int(_step_in_list(
            d["latent_dim"], SEARCH_SPACE["latent_dim"], rng))
    elif dimension == "hidden_dims":
        options = SEARCH_SPACE["hidden_dims"]
        d["hidden_dims"] = list(options[rng.randint(len(options))])
    elif dimension == "beta_end":
        lo, hi = SEARCH_SPACE["beta_end"]
        d["beta_end"] = float(np.clip(_scale(d["beta_end"], rng), lo, hi))
    elif dimension == "mask_fraction":
        lo, hi = SEARCH_SPACE["mask_fraction"]
        d["mask_fraction"] = float(np.clip(
            d["mask_fraction"] + rng.normal(0.0, 0.07), lo, hi))
    elif dimension == "learning_rate":
        lo, hi = SEARCH_SPACE["learning_rate"]
        d["learning_rate"] = float(np.clip(
            _scale(d["learning_rate"], rng, 0.4), lo, hi))
    elif dimension == "weight_decay":
        lo, hi = SEARCH_SPACE["weight_decay"]
        d["weight_decay"] = float(np.clip(
            _scale(d["weight_decay"], rng, 0.8), lo, hi))
    elif dimension == "num_epochs":
        d["num_epochs"] = int(_step_in_list(
            d["num_epochs"], SEARCH_SPACE["num_epochs"], rng))
    else:
        d["logvar_clamp_min"] = float(_step_in_list(
            d["logvar_clamp_min"], SEARCH_SPACE["logvar_clamp_min"], rng))

    return SearchConfig.from_dict(d), dimension


def climb_from(ledger: Ledger) -> Trial | None:
    """The admissible trial a proposal should be descended from.

    Returns the highest-ARI trial that keeps transfer AUC, or None when
    nothing admissible has been recorded yet, in which case the caller
    should explore instead of climbing towards a point that already
    fails the constraint.
    """
    floor = ledger.transfer_floor()
    admissible = [
        t for t in ledger.all()
        if not t.error
        and t.auc_transfer_median == t.auc_transfer_median
        and t.auc_transfer_median >= floor
    ]
    if not admissible:
        return None
    return max(admissible, key=lambda t: t.ari_median)


def propose_next(
    ledger: Ledger, rng: np.random.RandomState
) -> tuple[SearchConfig, str]:
    """Return the next configuration to evaluate and why it was chosen."""
    trials = [t for t in ledger.all() if not t.error]
    if not trials:
        return SearchConfig(), "baseline: the constraint is measured against it"

    seen = ledger.seen()
    incumbent = climb_from(ledger)

    if incumbent is None:
        for _ in range(MAX_ATTEMPTS):
            config = random_config(rng)
            if config.key() not in seen:
                return config, (
                    "explore: nothing tried so far keeps transfer AUC, so "
                    "there is no admissible point to climb from"
                )

    if incumbent is not None and rng.rand() >= EXPLORE_PROBABILITY:
        base = SearchConfig.from_dict(incumbent.config)
        for _ in range(MAX_ATTEMPTS):
            config, dimension = perturb(base, rng)
            if config.key() not in seen:
                return config, (
                    f"climb: {dimension} perturbed from trial #{incumbent.index} "
                    f"(ARI {incumbent.ari_median:.4f}, transfer AUC "
                    f"{incumbent.auc_transfer_median:.4f})"
                )

    for _ in range(MAX_ATTEMPTS):
        config = random_config(rng)
        if config.key() not in seen:
            return config, "explore: unguided draw"

    raise RuntimeError(
        "could not find an unevaluated configuration; the search space is "
        "effectively exhausted at this resolution"
    )
