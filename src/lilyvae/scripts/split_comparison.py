"""CLI: random row split against borehole split, encoder held fixed.

    uv run lilyvae-split-comparison --iterations 30

Writes data/results/split_comparison.json. Tests the claim in Section
4.4 that the borehole holdout costs nothing a random split does not.
"""

from __future__ import annotations

import argparse

import numpy as np

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.split_comparison import ARMS, run_split_comparison
from lilyvae.evaluation.targets import HEADLINE_TIERS
from lilyvae.scripts import write_result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--tiers", nargs="+", default=list(HEADLINE_TIERS))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_split_comparison(
        args.iterations, args.test_fraction, tuple(args.tiers)
    )

    print(f"\n{'target':<34}{'sample split':>14}{'borehole split':>16}")
    for row in result["summary"].values():
        print(f"{row['label'][:33]:<34}{row['sample_split']:>14.3f}"
              f"{row['borehole_split']:>16.3f}")
    meds = {
        arm: float(np.nanmedian([r[arm] for r in result["summary"].values()]))
        for arm in ARMS
    }
    print(f"{'median over targets':<34}{meds['sample_split']:>14.3f}"
          f"{meds['borehole_split']:>16.3f}")

    out = args.output or (RESULTS_DIR / "split_comparison.json")
    print(f"\nWritten to {write_result(out, result)}")


if __name__ == "__main__":
    main()
