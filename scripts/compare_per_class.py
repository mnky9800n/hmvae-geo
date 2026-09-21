"""Compare two configurations lithology by lithology, not on the average.

The macro AUC covers only about 45 of the 139 classes on a given fold,
because the rest lack two examples on one side of a borehole-level
split, and the covered set shifts from fold to fold. Two consequences
follow, and both matter for reading the search ledger.

A macro comparison can average over different class sets for the two
configurations, so the difference is partly a change of denominator.
This script fixes that by scoring only classes both configurations
covered on the same fold.

A configuration can also be much better on some lithologies and worse on
others while the average moves by nothing at all. That is invisible in
the ledger and is exactly what a single number is bad at.

Usage:
    uv run python scripts/compare_per_class.py \
        --a '{}' --b '{"beta_end": 0.3}' --folds 3
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from lilyvae.config import PROCESSED_DIR, RESULTS_DIR, SEED
from lilyvae.evaluation.folds import borehole_splits, nested_split
from lilyvae.evaluation.raw_baselines import evaluate_classification_transfer
from lilyvae.search.objective import _train
from lilyvae.search.space import SearchConfig
from lilyvae.training.trainer import get_device


def _per_class(config: SearchConfig, folds: int, device) -> dict[int, list[float]]:
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    bh = np.asarray(metadata["borehole_ids"])

    split = nested_split(bh, seed=SEED)
    f_pool, l_pool, bh_pool = features[split.search_mask], labels[split.search_mask], bh[split.search_mask]

    out: dict[int, list[float]] = {}
    for fold in borehole_splits(bh_pool, n_iterations=folds, test_fraction=0.1):
        vae = _train(config, f_pool[fold.train_mask], l_pool[fold.train_mask], device)
        vae.eval()
        with torch.no_grad():
            z_tr = vae.get_embeddings(f_pool[fold.train_mask].to(device)).cpu().numpy()
            z_te = vae.get_embeddings(f_pool[fold.test_mask].to(device)).cpu().numpy()
        res = evaluate_classification_transfer(
            z_tr, l_pool[fold.train_mask].numpy(),
            z_te, l_pool[fold.test_mask].numpy(),
        )
        for cls, auc in res["per_class_auc"].items():
            out.setdefault(int(cls), []).append(float(auc))
        print(f"  fold {fold.iteration}: macro {res['auc_macro']:.4f} "
              f"over {res['n_classes_evaluated']} classes")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="{}", help="JSON config A (reference)")
    ap.add_argument("--b", required=True, help="JSON config B")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dev = get_device(args.device)
    cfg_a = SearchConfig.from_dict(json.loads(args.a))
    cfg_b = SearchConfig.from_dict(json.loads(args.b))

    print(f"A = {json.dumps(cfg_a.to_dict())}")
    a = _per_class(cfg_a, args.folds, dev)
    print(f"B = {json.dumps(cfg_b.to_dict())}")
    b = _per_class(cfg_b, args.folds, dev)

    le = pickle.load(open(PROCESSED_DIR / "label_encoder.pkl", "rb"))
    counts = np.bincount(
        torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True).numpy()
    )

    # Only classes both configurations scored, so the comparison is not
    # partly a change of denominator.
    common = sorted(set(a) & set(b))
    rows = []
    for cls in common:
        ma, mb = float(np.median(a[cls])), float(np.median(b[cls]))
        rows.append((cls, le.classes_[cls], int(counts[cls]), ma, mb, mb - ma))

    rows.sort(key=lambda r: r[5])
    wins = sum(1 for r in rows if r[5] > 0.01)
    losses = sum(1 for r in rows if r[5] < -0.01)
    ties = len(rows) - wins - losses

    print(f"\n{len(common)} class(es) scored by both.")
    print(f"B better by >0.01 on {wins}, worse on {losses}, within 0.01 on {ties}.")
    print(f"macro over the common set: A {np.mean([r[3] for r in rows]):.4f}  "
          f"B {np.mean([r[4] for r in rows]):.4f}")

    print(f"\n{'lithology':<44} {'n':>7} {'A':>7} {'B':>7} {'B-A':>7}")
    print("-" * 78)
    for r in rows[:10]:
        print(f"{str(r[1])[:43]:<44} {r[2]:>7} {r[3]:>7.3f} {r[4]:>7.3f} {r[5]:>+7.3f}")
    print("  ...")
    for r in rows[-10:]:
        print(f"{str(r[1])[:43]:<44} {r[2]:>7} {r[3]:>7.3f} {r[4]:>7.3f} {r[5]:>+7.3f}")

    out = args.output or (RESULTS_DIR / "per_class_comparison.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config_a": cfg_a.to_dict(), "config_b": cfg_b.to_dict(),
            "n_folds": args.folds, "n_common_classes": len(common),
            "wins_b": wins, "losses_b": losses, "ties": ties,
            "per_class": [
                {"class_id": r[0], "lithology": str(r[1]), "n_samples": r[2],
                 "auc_a": r[3], "auc_b": r[4], "delta": r[5]} for r in rows
            ],
        }, f, indent=2)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
