"""Single-point full-data evaluation for baselines and ablation variants.

Complements evaluation/bootstrap.py, which bootstraps only the two full
VAE variants. Everything else (PCA, TabNet, SCARF, and the four
architectural ablation variants) is trained once on all 238K samples
and evaluated once on the same 238K samples. No folds, no iterations,
no CIs.

Zero-shot R^2 is added to each row by running CatBoost on the model's
embeddings for every held-out geophysical variable in the same
in-sample methodology that the paper uses for the full-data VAE.

Output JSON structure matches what the bootstrap module produces so
figure-generation and table-generation code can load either file
interchangeably:

    {
      "config": {...},
      "per_iteration": [ { "iteration": 0, "models": [...] } ],
      "summary": { model_name: { metric: { median, mean, ... } } },
      "elapsed_seconds": float
    }

For a single-point eval, "per_iteration" has one entry and all
`summary[model][metric]` aggregations collapse to the single value
(median == mean == ci_low == ci_high). The schema compatibility is
intentional so that a later merge step can stack VAE bootstrap rows
and baseline/ablation point rows into one table without branching.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from lilyvae.config import (
    PROCESSED_DIR,
    SEED,
)
from lilyvae.evaluation.bootstrap import (
    ALL_TRAINERS,
    TrainedModel,
    _aggregate_summary,
    _compute_metrics,
    _print_summary,
)
from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot
from lilyvae.training.trainer import get_device


#: Default set of models for single-point eval: everything that is NOT in
#: the bootstrap module's MODEL_REGISTRY. (The two VAE rows get both a
#: bootstrap and a single-point evaluation when the pipeline merges
#: results -- the bootstrap provides CIs and the single-point provides
#: the row used for zero-shot.)
SINGLE_POINT_MODELS: list[str] = [
    "pca",
    "tabnet_unsupervised",
    "scarf_contrastive",
    "ablation_fixed_variance",
    "ablation_zero_masking",
    "ablation_no_masking",
    "ablation_fixed_beta",
]


def run_single_point_eval(
    device: str = "auto",
    models: Iterable[str] | None = None,
    include_zeroshot: bool = True,
) -> dict:
    """Train each model once on full data and return metrics.

    Args:
        device: "auto", "cpu", "cuda", or "mps".
        models: Subset of ALL_TRAINERS names to run. Defaults to
            SINGLE_POINT_MODELS (everything except the two VAE variants
            that are bootstrapped separately).
        include_zeroshot: Whether to run CatBoost zero-shot prediction
            on each model's embeddings (adds ~30 s per model).

    Returns:
        A dict matching the bootstrap output schema: config,
        per_iteration (one entry), summary, elapsed_seconds.
    """
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)

    zeroshot_targets = None
    if include_zeroshot:
        zs_path = PROCESSED_DIR / "zeroshot_targets.pt"
        if zs_path.exists():
            zeroshot_targets = torch.load(zs_path, weights_only=False)

    wanted = set(models) if models is not None else set(SINGLE_POINT_MODELS)
    trainers = [(n, fn) for (n, fn) in ALL_TRAINERS if n in wanted]
    if not trainers:
        raise ValueError(
            f"No trainers matched {models}. Available: "
            f"{[n for n, _ in ALL_TRAINERS]}"
        )

    print(f"\n{'='*70}")
    print(f"Single-point full-data evaluation")
    print(f"  models: {[n for n, _ in trainers]}")
    print(f"  samples: {features.shape[0]}  include_zeroshot={include_zeroshot}")
    print(f"{'='*70}")

    start_time = time.time()
    model_results: list[dict] = []

    for name, trainer in trainers:
        t0 = time.time()
        print(f"\n  [{name}]")
        try:
            # Single-point eval == train on full data and evaluate on
            # that same full data. Passing features for both train and
            # test is deliberate: we want the paper's original
            # "in-sample on 238K samples" methodology, not a held-out
            # variant. This is the same codepath and assumptions the
            # paper has always used for the baseline and ablation rows.
            torch.manual_seed(SEED)
            trained: TrainedModel = trainer(features, labels, features, dev)
            metrics = _compute_metrics(trained, features, labels)

            # Zero-shot on the (in-sample) full-data embeddings
            if zeroshot_targets is not None:
                try:
                    zs_results = evaluate_all_zeroshot(
                        trained.embeddings, zeroshot_targets
                    )
                    zs_r2s = [
                        v["r2"] for v in zs_results.values()
                        if isinstance(v.get("r2"), float)
                        and not np.isnan(v["r2"])
                        and np.isfinite(v["r2"])
                    ]
                    if zs_r2s:
                        metrics["zeroshot_r2_avg"] = float(np.mean(zs_r2s))
                        metrics["zeroshot_n_variables"] = len(zs_r2s)
                except Exception as e:
                    print(f"    zeroshot ERROR: {e}")

            zs = metrics.get("zeroshot_r2_avg", float("nan"))
            print(
                f"    R2={metrics['r2_avg']:.4f}  ARI={metrics['ari']:.4f}  "
                f"AUC={metrics['auc_macro']:.4f}  ZS={zs:.4f}  "
                f"({time.time()-t0:.1f}s)"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            metrics = {"model": name, "error": str(e)}

        model_results.append(metrics)

    elapsed = time.time() - start_time

    # Wrap in the bootstrap output schema so downstream code can load
    # either a bootstrap result or a single-point result without branching.
    per_iteration = [{
        "iteration": 0,
        "n_train_samples": int(features.shape[0]),
        "n_test_samples": int(features.shape[0]),
        "n_test_boreholes": 0,  # no split
        "models": model_results,
        "elapsed_seconds": elapsed,
    }]

    summary = _aggregate_summary(per_iteration)

    print(f"\n{'='*70}")
    print(f"Single-point eval complete in {elapsed/60:.1f} min")
    print(f"{'='*70}")
    _print_summary(summary)

    return {
        "config": {
            "mode": "single_point_full_data",
            "models": [n for n, _ in trainers],
            "include_zeroshot": include_zeroshot,
            "seed": SEED,
            "n_samples": int(features.shape[0]),
        },
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }
