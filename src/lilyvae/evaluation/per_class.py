"""Does supervision help any individual lithology?

The macro figures say the two models are indistinguishable, and a macro
figure can hide a configuration that is much better on some classes and
worse on others. If the labels carried information the measurements do
not, the gain should appear on the classes the measurements separate
least well, so the comparison worth making is per class against the
unsupervised model's own score on that class.

Scored on held-out boreholes across the shared folds, and only on
classes both models could score on the same fold, so a difference is
not partly a change of denominator.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lilyvae.config import PROCESSED_DIR, SEED
from lilyvae.evaluation.bootstrap import ALL_TRAINERS
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.evaluation.raw_baselines import evaluate_classification_transfer
from lilyvae.training.trainer import get_device


def run_per_class(
    n_iterations: int = 5, test_fraction: float = 0.1, device: str = "auto"
) -> dict:
    """Per-class transfer AUC for the unsupervised and semi-supervised models."""
    dev = get_device(device)
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])
    trainers = dict(ALL_TRAINERS)

    per_class: dict[str, dict[int, list[float]]] = {
        "unsupervised_vae": {}, "semi_supervised_vae": {}
    }
    start = time.time()

    for fold in borehole_splits(borehole_ids, n_iterations, test_fraction, seed=SEED):
        print(f"\nFold {fold.iteration + 1}/{n_iterations}", flush=True)
        fold_scores = {}
        for name in per_class:
            trained = trainers[name](
                features[fold.train_mask], labels[fold.train_mask], features, dev
            )
            z = trained.embeddings.numpy()
            res = evaluate_classification_transfer(
                z[fold.train_mask], labels[fold.train_mask].numpy(),
                z[fold.test_mask], labels[fold.test_mask].numpy(),
            )
            fold_scores[name] = res["per_class_auc"]
            print(f"  {name:<22} macro {res['auc_macro']:.4f} "
                  f"over {res['n_classes_evaluated']} classes", flush=True)

        # Only classes both models scored on this fold.
        shared = set(fold_scores["unsupervised_vae"]) & set(fold_scores["semi_supervised_vae"])
        for name in per_class:
            for cls in shared:
                per_class[name].setdefault(int(cls), []).append(
                    float(fold_scores[name][cls])
                )

    counts = np.bincount(labels.numpy())
    rows = []
    for cls in sorted(per_class["unsupervised_vae"]):
        u = float(np.median(per_class["unsupervised_vae"][cls]))
        s = float(np.median(per_class["semi_supervised_vae"][cls]))
        rows.append({
            "class_id": cls, "n_samples": int(counts[cls]),
            "auc_unsupervised": u, "auc_semi_supervised": s, "delta": s - u,
            "n_folds": len(per_class["unsupervised_vae"][cls]),
        })

    deltas = np.array([r["delta"] for r in rows])
    base = np.array([r["auc_unsupervised"] for r in rows])
    correlation = (
        float(np.corrcoef(base, deltas)[0, 1]) if len(rows) > 2 else float("nan")
    )

    return {
        "config": {"n_iterations": n_iterations, "seed": SEED},
        "per_class": rows,
        "summary": {
            "n_classes": len(rows),
            "semi_better_by_0.01": int((deltas > 0.01).sum()),
            "unsup_better_by_0.01": int((deltas < -0.01).sum()),
            "within_0.01": int((np.abs(deltas) <= 0.01).sum()),
            "median_delta": float(np.median(deltas)),
            # If labels carried extra information it should show up most
            # where the measurements do worst, i.e. a negative correlation
            # between the unsupervised score and the gain.
            "corr_delta_vs_unsupervised_auc": correlation,
        },
        "elapsed_seconds": time.time() - start,
    }
