"""CLI: test the literature's predictions against the sparse features.

    uv run lilyvae-sae-hypotheses --folds 3 --null 30
"""

from __future__ import annotations

import argparse
import json

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.sae_hypotheses import run_hypothesis_test
from lilyvae.scripts import make_serializable


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--null", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_hypothesis_test(args.folds, args.hidden, args.null, args.device)

    print(f"\n{'':>18}{'--- best correlation ---':>34}"
          f"{'--- selectivity ---':>30}")
    print(f"{'hypothesis':<18}{'obs':>9}{'null':>9}{'p95':>9}"
          f"{'obs':>10}{'null':>9}{'p95':>9}   verdict")
    print("-" * 92)
    for r in result["results"]:
        flags = []
        if r["supported"]:
            flags.append("correlated")
        if r.get("selective"):
            flags.append("selective")
        print(f"{r['hypothesis']:<18}{r['observed_rho']:>9.3f}"
              f"{r['null_rho']:>9.3f}{r['null_p95']:>9.3f}"
              f"{r.get('selectivity', float('nan')):>10.3f}"
              f"{r.get('selectivity_null', float('nan')):>9.3f}"
              f"{r.get('selectivity_null_p95', float('nan')):>9.3f}"
              f"   {' + '.join(flags) or 'neither'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (RESULTS_DIR / "sae_hypotheses.json")
    with open(out, "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
