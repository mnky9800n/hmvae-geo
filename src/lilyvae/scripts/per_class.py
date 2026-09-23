"""CLI: per-lithology comparison of the two full models.

    uv run lilyvae-per-class --iterations 5
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.per_class import run_per_class
from lilyvae.scripts import make_serializable


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_per_class(args.iterations, args.test_fraction, args.device)
    s = result["summary"]
    print(f"\n{s['n_classes']} classes scored by both models")
    print(f"  semi-supervised better by >0.01 on {s['semi_better_by_0.01']}")
    print(f"  unsupervised better by >0.01 on   {s['unsup_better_by_0.01']}")
    print(f"  within 0.01 on                    {s['within_0.01']}")
    print(f"  median difference                 {s['median_delta']:+.4f}")
    print(f"  correlation of gain with the unsupervised score "
          f"{s['corr_delta_vs_unsupervised_auc']:+.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (RESULTS_DIR / "per_class.json")
    with open(out, "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
