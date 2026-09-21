"""What the search is allowed to vary.

Kept deliberately small. Every dimension added multiplies the number of
configurations that have to be tried before a reported improvement means
anything, and each trial is a full retrain across several folds.

Two knobs that look tempting are excluded on purpose.

``n_gmm_components`` is not searched. ARI is measured by fitting a
Gaussian mixture to the embedding and comparing its partition to the
lithology labels, so raising the component count moves ARI by changing
the metric rather than the representation. It stays pinned at the value
the paper reports.

``alpha`` is not searched above zero by default. Supervision weight is
the subject of the paper's central claim, not a knob to tune: the result
is that raising it buys nothing. Tuning it to maximise label recovery
would answer a different question from the one being asked. It is
available for an explicit sweep through `lilyvae-train`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from lilyvae.config import (
    BETA_END,
    HIDDEN_DIMS,
    LATENT_DIM,
    LEARNING_RATE,
    MASK_FRACTION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)


@dataclass(frozen=True)
class SearchConfig:
    """One point in the search space."""

    latent_dim: int = LATENT_DIM
    hidden_dims: tuple[int, ...] = tuple(HIDDEN_DIMS)
    beta_end: float = BETA_END
    mask_fraction: float = MASK_FRACTION
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_epochs: int = NUM_EPOCHS
    logvar_clamp_min: float = -6.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SearchConfig":
        d = dict(d)
        if "hidden_dims" in d:
            d["hidden_dims"] = tuple(d["hidden_dims"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown config keys {sorted(unknown)}; allowed: {sorted(known)}"
            )
        return cls(**d)

    def key(self) -> str:
        """Stable identity, so a repeated configuration is not re-run."""
        return repr(tuple(sorted(self.to_dict().items())))


#: Ranges the random sampler draws from. The agent loop is free to
#: propose values outside these, which is much of the point of having an
#: agent rather than a sampler, but these bound the unguided fallback.
SEARCH_SPACE: dict[str, object] = {
    "latent_dim": [4, 6, 8, 10, 12, 16, 24, 32],
    "hidden_dims": [
        (64, 32), (128, 64), (256, 128), (128, 128), (256, 128, 64),
    ],
    "beta_end": (1e-4, 1.0),          # log-uniform
    "mask_fraction": (0.0, 0.5),      # uniform
    "learning_rate": (1e-4, 5e-3),    # log-uniform
    "weight_decay": (1e-7, 1e-3),     # log-uniform
    "num_epochs": [50, 100, 150],
    "logvar_clamp_min": [-12.0, -6.0, -3.0],
}


def random_config(rng: np.random.RandomState) -> SearchConfig:
    """Draw an unguided configuration, for seeding the ledger."""

    def loguniform(lo: float, hi: float) -> float:
        return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))

    return SearchConfig(
        latent_dim=int(rng.choice(SEARCH_SPACE["latent_dim"])),
        hidden_dims=tuple(
            SEARCH_SPACE["hidden_dims"][rng.randint(len(SEARCH_SPACE["hidden_dims"]))]
        ),
        beta_end=loguniform(*SEARCH_SPACE["beta_end"]),
        mask_fraction=float(rng.uniform(*SEARCH_SPACE["mask_fraction"])),
        learning_rate=loguniform(*SEARCH_SPACE["learning_rate"]),
        weight_decay=loguniform(*SEARCH_SPACE["weight_decay"]),
        num_epochs=int(rng.choice(SEARCH_SPACE["num_epochs"])),
        logvar_clamp_min=float(rng.choice(SEARCH_SPACE["logvar_clamp_min"])),
    )
