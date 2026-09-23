"""Score one configuration on the search boreholes.

The search never touches the reporting pool. `nested_split` partitions
boreholes once; folds are drawn inside the search pool only, and the
reporting boreholes are scored at the very end by
`lilyvae-search report`, for the single configuration the search
settled on.

ARI is measured exactly as the paper measures it, with the Gaussian
mixture component count pinned, so a configuration cannot raise its
score by changing the metric.

AUC here is the linear probe, fitted and scored on the fold's held-out
embeddings, matching what `evaluation/bootstrap.py` reports. It measures
how separable lithology is in the representation. The transfer AUC, with
the probe fitted on training boreholes, is also recorded, because a
configuration that improves separability without improving transfer has
not produced a more useful representation.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lilyvae.config import MAX_GRAD_NORM, PROCESSED_DIR, SEED
from lilyvae.evaluation.ablations import _train_ablation_model
from lilyvae.evaluation.bootstrap import _make_vae_dataloader
from lilyvae.evaluation.classification import evaluate_classification
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.folds import borehole_splits, nested_split
from lilyvae.model.vae import LilyVAE
from lilyvae.search.space import SearchConfig
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.trainer import get_device
from sklearn.metrics import r2_score


def _load():
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    return features, labels, np.asarray(metadata["borehole_ids"])


def _train(config: SearchConfig, features, labels, device) -> LilyVAE:
    torch.manual_seed(SEED)
    vae = LilyVAE(
        input_dim=features.shape[1],
        latent_dim=config.latent_dim,
        hidden_dims=list(config.hidden_dims),
        logvar_clamp=(config.logvar_clamp_min, 6.0),
    )
    masking = LearnableMasking(features.shape[1], config.mask_fraction)
    return _train_ablation_model(
        dataloader=_make_vae_dataloader(features, labels),
        vae=vae,
        masking=masking,
        decoder_type="heteroscedastic",
        beta_schedule_type="annealing",
        alpha_weight=0.0,
        num_epochs=config.num_epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
        beta_end=config.beta_end,
    )


def evaluate_config(
    config: SearchConfig,
    n_folds: int = 3,
    device: str = "auto",
    pool: str = "search",
    report_fraction: float = 0.2,
) -> dict:
    """Train and score one configuration across folds of one pool.

    Args:
        config: The configuration to evaluate.
        n_folds: Folds to average over. Three is enough to rank
            candidates while keeping a trial affordable; the final
            configuration should be re-scored with more.
        device: "auto", "cpu", "cuda", or "mps".
        pool: "search" to use the tuning boreholes, "report" to score the
            reserved ones. Only call with "report" once, at the end.
        report_fraction: Fraction of boreholes reserved from the search.

    Returns:
        dict of median and interquartile range for ARI, probe AUC,
        transfer AUC and reconstruction R^2, plus the per-fold values.
    """
    features, labels, borehole_ids = _load()
    dev = get_device(device)

    split = nested_split(borehole_ids, report_fraction=report_fraction, seed=SEED)
    mask = split.search_mask if pool == "search" else split.report_mask
    pool_features = features[mask]
    pool_labels = labels[mask]
    pool_boreholes = borehole_ids[mask]

    per_fold = []
    start = time.time()

    for fold in borehole_splits(pool_boreholes, n_iterations=n_folds, test_fraction=0.1):
        train_f = pool_features[fold.train_mask]
        train_l = pool_labels[fold.train_mask]
        test_f = pool_features[fold.test_mask]
        test_l = pool_labels[fold.test_mask]

        vae = _train(config, train_f, train_l, dev)
        vae.eval()
        with torch.no_grad():
            z_test = vae.get_embeddings(test_f.to(dev)).cpu()
            z_train = vae.get_embeddings(train_f.to(dev)).cpu()
            recon = vae(test_f.to(dev))["x_mu"].cpu().numpy()

        entry = {
            "ari": float(evaluate_clustering(z_test, test_l)["ari"]),
            "auc_probe": float(
                evaluate_classification(z_test, test_l)["auc_macro"]
            ),
            "r2": float(np.mean([
                r2_score(test_f.numpy()[:, i], recon[:, i])
                for i in range(test_f.shape[1])
            ])),
        }
        # Transfer AUC: probe fitted on training boreholes only.
        from lilyvae.evaluation.raw_baselines import evaluate_classification_transfer
        cls = evaluate_classification_transfer(
            z_train.numpy(), train_l.numpy(),
            z_test.numpy(), test_l.numpy(),
        )
        entry["auc_transfer"] = float(cls["auc_macro"])
        # Keep the per-class detail. The macro average covers only about
        # 45 of the 139 classes on any fold, because the rest lack two
        # examples on one side of the split, and the covered set changes
        # from fold to fold. Comparing two configurations on the macro
        # number alone can therefore compare averages over different
        # class sets, and it hides a configuration that is much better on
        # some lithologies and worse on others.
        entry["n_classes"] = int(cls["n_classes_evaluated"])
        entry["per_class_transfer"] = {
            str(k): float(v) for k, v in cls["per_class_auc"].items()
        }
        per_fold.append(entry)

    def agg(key: str) -> tuple[float, list[float]]:
        vals = np.array([f[key] for f in per_fold], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), [float("nan"), float("nan")]
        return float(np.median(vals)), [
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)),
        ]

    ari_m, ari_i = agg("ari")
    auc_m, auc_i = agg("auc_probe")
    tra_m, tra_i = agg("auc_transfer")
    r2_m, _ = agg("r2")

    # Median per class across the folds where that class was scorable,
    # with the fold count kept so a class seen once is not read as
    # confidently as one seen five times.
    by_class: dict[str, list[float]] = {}
    for f in per_fold:
        for cls_id, auc in f.get("per_class_transfer", {}).items():
            by_class.setdefault(cls_id, []).append(auc)
    per_class_transfer = {
        k: {
            "auc_median": float(np.median(v)),
            "n_folds": len(v),
            # Fold spread for this class, so an apparent per-class gain
            # can be compared against how much that class moves anyway.
            "auc_sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "folds": [float(x) for x in v],
        }
        for k, v in sorted(by_class.items(), key=lambda kv: int(kv[0]))
    }

    return {
        "pool": pool,
        "n_folds": len(per_fold),
        "per_class_transfer": per_class_transfer,
        "n_classes_scored": len(per_class_transfer),
        "ari_median": ari_m, "ari_iqr": ari_i,
        "auc_median": auc_m, "auc_iqr": auc_i,
        "auc_transfer_median": tra_m, "auc_transfer_iqr": tra_i,
        "r2_median": r2_m,
        "per_fold": per_fold,
        "elapsed_seconds": time.time() - start,
    }
