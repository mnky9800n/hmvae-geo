"""CLI: what depth explains inside one borehole, and pooled across them.

    uv run lilyvae-depth-within

Writes data/results/depth_within.json. The third number in
Figure~\\ref{fig:depth}, depth on a borehole the fit has never seen,
comes from `lilyvae-depth-bootstrap` instead.
"""

from __future__ import annotations

import argparse

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.depth_within import run_depth_within
from lilyvae.evaluation.targets import HEADLINE_TIERS
from lilyvae.scripts import write_result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tiers", nargs="+", default=list(HEADLINE_TIERS))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_depth_within(tuple(args.tiers))
    s = result["summary"]
    print(f"\nMedian over {s['n_targets']} targets: "
          f"within borehole {s['within_borehole_median']:.3f}, "
          f"pooled {s['pooled_median']:.3f}")

    out = args.output or (RESULTS_DIR / "depth_within.json")
    print(f"\nWritten to {write_result(out, result)}")


if __name__ == "__main__":
    main()
