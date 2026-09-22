"""What does an energy distance of 0.066 actually mean?

The statistic has no natural scale, so a value is uninterpretable until
it is compared to something. This applies a known perturbation to a
random tenth of the rows and reports the resulting distance, which turns
the axis of `fig_holdout_novelty.png` into a physical statement.

The distance grows roughly as the square of the displacement, so the
median held-out fold at 0.066 corresponds to displacing one standardised
input variable by about 0.45 standard deviations.
"""

from __future__ import annotations

import numpy as np
import torch

from lilyvae.config import PROCESSED_DIR, SEED
from lilyvae.evaluation.folds import Fold, energy_distance


def main() -> None:
    x = torch.load(
        PROCESSED_DIR / "train_features.pt", weights_only=True
    ).numpy().astype(np.float64)
    rng = np.random.RandomState(SEED)
    n = len(x)

    def distance(column: int | None = None, shift: float = 0.0,
                 scale: float | None = None) -> float:
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, n // 10, replace=False)] = True
        y = x.copy()
        if shift:
            y[mask, column] += shift
        if scale:
            y[mask] *= scale
        fold = Fold(0, ~mask, mask, np.array([]), np.array([]))
        return energy_distance(fold, y)

    print(f"{'perturbation applied to a random tenth of the rows':<48} {'energy':>8}")
    print("-" * 58)
    print(f"{'none (random split, no shift)':<48} {distance():>8.4f}")
    for sd in (0.25, 0.5, 1.0, 2.0):
        print(f"{f'bulk density shifted by {sd:.2f} SD':<48} "
              f"{distance(column=0, shift=sd):>8.4f}")
    print(f"{'all six variables inflated 1.5x in spread':<48} "
          f"{distance(scale=1.5):>8.4f}")


if __name__ == "__main__":
    main()
