"""CLI entry point for ablation studies and baseline comparisons.

Usage
-----
Run everything:
    lilyvae-ablation --experiment all

Run only the decoder ablation:
    lilyvae-ablation --experiment decoder_type

Run only baselines:
    lilyvae-ablation --experiment baselines

All results are written to data/results/ablations.json. Subsequent runs
merge into the existing file so that partial runs can be resumed and
individual experiments can be re-run without discarding prior work.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from lilyvae.config import RESULTS_DIR, NUM_EPOCHS, LATENT_DIM
from lilyvae.evaluation.ablations import ABLATION_GRID, run_ablation, run_all_ablations
from lilyvae.evaluation.baselines import run_all_baselines


# ---------------------------------------------------------------------------
# Serialization helper (shared with scripts/evaluate.py by convention)
# ---------------------------------------------------------------------------


def _make_serializable(obj):
    """Recursively convert numpy scalars and arrays to Python builtins."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _load_existing_results(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def _run_single_ablation(
    ablation_name: str,
    results: dict,
    num_epochs: int,
    device: str,
) -> None:
    """Run one ablation group, updating results in-place."""
    values = ABLATION_GRID[ablation_name]
    ablation_results = []

    for value in values:
        result = run_ablation(
            ablation_name,
            value,
            num_epochs=num_epochs,
            device=device,
        )
        ablation_results.append(result)

        # Print a brief summary after each value so the user gets feedback
        # during long runs without waiting for the full group to finish.
        print(
            f"  -> r2_avg={result.get('r2_avg', float('nan')):.4f}  "
            f"ari={result.get('ari', float('nan')):.4f}  "
            f"auc_macro={result.get('auc_macro', float('nan')):.4f}"
        )

    results.setdefault("ablations", {})[ablation_name] = ablation_results


def _run_baselines(results: dict, num_epochs: int, device: str) -> None:
    """Run all baselines and record results."""
    baseline_results = run_all_baselines(
        latent_dim=LATENT_DIM,
        num_epochs=num_epochs,
        device=device,
    )
    for r in baseline_results:
        print(
            f"  {r['model']:35s}  "
            f"r2_avg={r.get('r2_avg', float('nan')):.4f}  "
            f"ari={r.get('ari', float('nan')):.4f}  "
            f"auc_macro={r.get('auc_macro', float('nan')):.4f}"
        )
    results["baselines"] = baseline_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    valid_experiments = ["all", "baselines"] + list(ABLATION_GRID.keys())

    parser = argparse.ArgumentParser(
        description="Run LilyVAE ablation studies and baseline comparisons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "experiments:",
            "  all              Run every ablation and every baseline",
            "  baselines        Run PCA, linear AE, and deterministic AE",
            *[f"  {k:<16} Run the {k!r} ablation" for k in ABLATION_GRID],
        ]),
    )
    parser.add_argument(
        "--experiment",
        choices=valid_experiments,
        default="all",
        metavar="EXPERIMENT",
        help=(
            "Which experiment to run. "
            f"Choices: {', '.join(valid_experiments)}. "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        metavar="N",
        help=f"Training epochs for each model (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "ablations.json",
        metavar="PATH",
        help="Output JSON path (default: data/results/ablations.json)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results = _load_existing_results(args.output)

    print("=" * 60)
    print(f"LilyVAE ablation runner")
    print(f"  experiment : {args.experiment}")
    print(f"  epochs     : {args.epochs}")
    print(f"  device     : {args.device}")
    print(f"  output     : {args.output}")
    print("=" * 60)

    if args.experiment == "all":
        for ablation_name in ABLATION_GRID:
            print(f"\n{'='*60}")
            print(f"Ablation: {ablation_name}")
            print("=" * 60)
            _run_single_ablation(ablation_name, results, args.epochs, args.device)

        print(f"\n{'='*60}")
        print("Baselines")
        print("=" * 60)
        _run_baselines(results, args.epochs, args.device)

    elif args.experiment == "baselines":
        print(f"\n{'='*60}")
        print("Baselines")
        print("=" * 60)
        _run_baselines(results, args.epochs, args.device)

    else:
        # Single ablation group
        ablation_name = args.experiment
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print("=" * 60)
        _run_single_ablation(ablation_name, results, args.epochs, args.device)

    _save_results(results, args.output)


if __name__ == "__main__":
    main()
