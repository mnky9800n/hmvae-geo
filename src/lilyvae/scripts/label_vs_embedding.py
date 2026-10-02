"""CLI: does the lithology label predict what the embedding predicts?

    uv run lilyvae-label-vs-embedding --iterations 30

Writes data/results/label_vs_embedding.json. Tests the Discussion's
claim that categorisation is a lower-fidelity record of process than the
instruments on the same core.
"""

from __future__ import annotations

import argparse

from lilyvae.config import RESULTS_DIR
from lilyvae.evaluation.label_vs_embedding import ARMS, run_label_vs_embedding
from lilyvae.evaluation.targets import HEADLINE_TIERS
from lilyvae.scripts import write_result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--tiers", nargs="+", default=list(HEADLINE_TIERS))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_label_vs_embedding(
        args.iterations, args.test_fraction, tuple(args.tiers)
    )

    print(f"\n{'target':<32}" + "".join(f"{a:>18}" for a in ARMS))
    for row in result["summary"].values():
        print(f"{row['label_name'][:31]:<32}"
              + "".join(f"{row[a]:>18.3f}" for a in ARMS))
    print(f"{'median over targets':<32}"
          + "".join(f"{result['overall'][a]:>18.3f}" for a in ARMS))

    out = args.output or (RESULTS_DIR / "label_vs_embedding.json")
    print(f"\nWritten to {write_result(out, result)}")


if __name__ == "__main__":
    main()
