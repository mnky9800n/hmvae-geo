"""CLI for the depth-confound bootstrap and the depth-as-input ablation.

    uv run lilyvae-depth-bootstrap                 # the four arms, 30 folds
    uv run lilyvae-depth-bootstrap --ablation      # depth as a 7th input

Writes data/results/depth_bootstrap.json and
data/results/depth_input_ablation.json.
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.depth_bootstrap import (
    run_depth_bootstrap,
    run_depth_input_ablation,
)
from lilyvae.evaluation.targets import HEADLINE_TIERS
from lilyvae.scripts import write_result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--tiers", nargs="+", default=list(HEADLINE_TIERS))
    p.add_argument("--device", default="auto")
    p.add_argument("--ablation", action="store_true",
                   help="run the depth-as-input ablation instead")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.ablation:
        result = run_depth_input_ablation(
            n_iterations=args.iterations, test_fraction=args.test_fraction,
            tiers=tuple(args.tiers), device=args.device,
        )
        default = RESULTS_DIR / "depth_input_ablation.json"
    else:
        result = run_depth_bootstrap(
            n_iterations=args.iterations, test_fraction=args.test_fraction,
            tiers=tuple(args.tiers), device=args.device,
        )
        default = RESULTS_DIR / "depth_bootstrap.json"

    out = args.output or default
    write_result(out, result)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
