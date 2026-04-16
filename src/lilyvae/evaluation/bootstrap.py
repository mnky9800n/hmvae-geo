"""Bootstrap training-stability evaluation for the VAE variants.

This module measures how the VAE's reconstruction R^2, clustering ARI,
and classification AUC vary across different train/test borehole splits.
Borehole-level splits are the right granularity because individual
boreholes are geologically non-exchangeable (they sample distinct
ocean basins, depths, and sediment histories), so a random sample-level
split would leak information across splits.

What is and isn't in the registry
---------------------------------
Only the two full VAE variants (unsupervised, semi-supervised) are
bootstrapped, because the paper's quantitative claim is about *those*
models' stability under retraining. Baselines (PCA, TabNet, SCARF) and
ablation variants (fixed-variance decoder, zero masking, no masking,
fixed beta) are evaluated as single-point full-data runs in a separate
codepath (`evaluation/single_point.py`). Bootstrapping them was folly
-- PCA is deterministic and TabNet/SCARF's CIs aren't part of the
story the paper is telling.

Zero-shot R^2 is also NOT in this module. It is a property of the
full-data trained model's embedding space and is computed once via
`evaluate_all_zeroshot` on the embeddings saved by training.

For each bootstrap iteration this module:

1. Splits the 296 boreholes into 90% train / 10% held-out (seeded).
2. Trains the unsupervised VAE AND the semi-supervised VAE from scratch
   on the 90% training boreholes.
3. Evaluates both on the held-out boreholes: reconstruction R^2, ARI,
   AUC.
4. Records per-iteration metrics.

After N iterations we report median / IQR / 95% CI per model per metric.
The trainer and metric-computation helpers for the non-bootstrapped
models (baselines, ablations) live in this file because they are reused
by the single-point evaluator, but they are not referenced from
MODEL_REGISTRY.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from lilyvae.config import (
    ALPHA_END,
    BATCH_SIZE,
    HIDDEN_DIMS,
    INPUT_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MASK_FRACTION,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    PROCESSED_DIR,
    SEED,
    TRAINING_VARIABLES,
    WEIGHT_DECAY,
)
from lilyvae.evaluation.ablations import (
    FixedVarianceVAE,
    NoMasking,
    ZeroMasking,
    _train_ablation_model,
)
from lilyvae.evaluation.baselines import (
    PCABaseline,
    SCARFBaseline,
    TabNetBaseline,
    _train_scarf,
)
from lilyvae.evaluation.classification import evaluate_classification
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.model.vae import LilyVAE
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.trainer import get_device


# ===========================================================================
# Model adapter: common fit / get_embeddings / reconstruction interface
# ===========================================================================


@dataclass
class TrainedModel:
    """Uniform wrapper over heterogeneous trained models.

    Every bootstrap slot produces one of these. Downstream evaluation code
    only needs: embeddings on test set, optional reconstruction on test set.
    """

    name: str
    embeddings: torch.Tensor  # (N_test, latent_dim)
    reconstruction: np.ndarray | None = None  # (N_test, 6) or None if N/A


# A Trainer takes (train_features, train_labels, test_features, device) and
# returns a TrainedModel with embeddings + reconstruction evaluated on
# test_features.
Trainer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.device], TrainedModel]


def _make_vae_dataloader(features: torch.Tensor, labels: torch.Tensor) -> DataLoader:
    dataset = TensorDataset(features, labels, torch.arange(len(features)))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# ---------------------------------------------------------------------------
# VAE + ablation trainers (delegated to ablations.py's _train_ablation_model)
# ---------------------------------------------------------------------------


def _run_vae(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    device: torch.device,
    *,
    name: str,
    vae_factory: Callable[[], nn.Module],
    masking_factory: Callable[[], nn.Module],
    decoder_type: str,
    beta_type: str,  # "annealing" or "fixed"
    alpha_weight: float,  # 0.0 for unsupervised
) -> TrainedModel:
    torch.manual_seed(SEED)

    vae = vae_factory()
    masking = masking_factory()
    dataloader = _make_vae_dataloader(train_features, train_labels)

    vae = _train_ablation_model(
        dataloader=dataloader,
        vae=vae,
        masking=masking,
        decoder_type=decoder_type,
        beta_schedule_type=beta_type,
        alpha_weight=alpha_weight,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
    )

    vae.eval()
    with torch.no_grad():
        x = test_features.to(device)
        out = vae(x)
        embeddings = vae.get_embeddings(x).cpu()
        reconstruction = out["x_mu"].cpu().numpy()

    return TrainedModel(name=name, embeddings=embeddings, reconstruction=reconstruction)


def _trainer_unsupervised_vae(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="unsupervised_vae",
        vae_factory=lambda: LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: LearnableMasking(INPUT_DIM, MASK_FRACTION),
        decoder_type="heteroscedastic",
        beta_type="annealing",
        alpha_weight=0.0,
    )


def _trainer_semi_supervised_vae(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="semi_supervised_vae",
        vae_factory=lambda: LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: LearnableMasking(INPUT_DIM, MASK_FRACTION),
        decoder_type="heteroscedastic",
        beta_type="annealing",
        alpha_weight=ALPHA_END,
    )


def _trainer_ablation_fixed_variance(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="ablation_fixed_variance",
        vae_factory=lambda: FixedVarianceVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: LearnableMasking(INPUT_DIM, MASK_FRACTION),
        decoder_type="fixed_variance",
        beta_type="annealing",
        alpha_weight=0.0,
    )


def _trainer_ablation_zero_masking(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="ablation_zero_masking",
        vae_factory=lambda: LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: ZeroMasking(MASK_FRACTION),
        decoder_type="heteroscedastic",
        beta_type="annealing",
        alpha_weight=0.0,
    )


def _trainer_ablation_no_masking(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="ablation_no_masking",
        vae_factory=lambda: LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: NoMasking(),
        decoder_type="heteroscedastic",
        beta_type="annealing",
        alpha_weight=0.0,
    )


def _trainer_ablation_fixed_beta(train_f, train_l, test_f, device):
    return _run_vae(
        train_f, train_l, test_f, device,
        name="ablation_fixed_beta",
        vae_factory=lambda: LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)),
        masking_factory=lambda: LearnableMasking(INPUT_DIM, MASK_FRACTION),
        decoder_type="heteroscedastic",
        beta_type="fixed",
        alpha_weight=0.0,
    )


# ---------------------------------------------------------------------------
# Baseline trainers
# ---------------------------------------------------------------------------


def _trainer_pca(train_f, train_l, test_f, device):
    torch.manual_seed(SEED)
    baseline = PCABaseline(LATENT_DIM)
    dataloader = _make_vae_dataloader(train_f, train_l)
    baseline.fit(dataloader)
    embeddings = baseline.get_embeddings(test_f)
    out = baseline(test_f)
    return TrainedModel(
        name="pca",
        embeddings=embeddings,
        reconstruction=out["x_mu"].numpy(),
    )



def _trainer_tabnet(train_f, train_l, test_f, device):
    torch.manual_seed(SEED)
    baseline = TabNetBaseline(LATENT_DIM)
    dataloader = _make_vae_dataloader(train_f, train_l)
    baseline.fit(dataloader, max_epochs=NUM_EPOCHS)
    embeddings = baseline.get_embeddings(test_f)
    return TrainedModel(
        name="tabnet_unsupervised",
        embeddings=embeddings,
        reconstruction=None,  # masked-prediction, not full reconstruction
    )


def _trainer_scarf(train_f, train_l, test_f, device):
    torch.manual_seed(SEED)
    model = SCARFBaseline(INPUT_DIM, LATENT_DIM)
    dataloader = _make_vae_dataloader(train_f, train_l)
    model = _train_scarf(
        model, dataloader, NUM_EPOCHS, LEARNING_RATE,
        WEIGHT_DECAY, MAX_GRAD_NORM, device,
    )
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(test_f.to(device)).cpu()
    return TrainedModel(
        name="scarf_contrastive",
        embeddings=embeddings,
        reconstruction=None,  # contrastive, no decoder
    )


# ---------------------------------------------------------------------------
# Model registry (order = reporting order in the output file)
# ---------------------------------------------------------------------------


#: The bootstrap only touches the two full VAE variants. See the module
#: docstring for the rationale. Baselines and ablations are evaluated
#: single-point in evaluation/single_point.py, but their trainer functions
#: live in this file so both evaluators share one source of truth.
MODEL_REGISTRY: list[tuple[str, Trainer]] = [
    ("semi_supervised_vae", _trainer_semi_supervised_vae),
    ("unsupervised_vae", _trainer_unsupervised_vae),
]


#: Full list of trainers (including baselines + ablations) for the
#: single-point evaluator to reuse.
ALL_TRAINERS: list[tuple[str, Trainer]] = [
    # Baselines
    ("pca", _trainer_pca),
    ("tabnet_unsupervised", _trainer_tabnet),
    ("scarf_contrastive", _trainer_scarf),
    # Ablations (VAE with one component changed from default)
    ("ablation_fixed_variance", _trainer_ablation_fixed_variance),
    ("ablation_zero_masking", _trainer_ablation_zero_masking),
    ("ablation_no_masking", _trainer_ablation_no_masking),
    ("ablation_fixed_beta", _trainer_ablation_fixed_beta),
    # Full models
    ("semi_supervised_vae", _trainer_semi_supervised_vae),
    ("unsupervised_vae", _trainer_unsupervised_vae),
]


# ===========================================================================
# Per-model metric computation
# ===========================================================================


def _compute_metrics(
    trained: TrainedModel,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict:
    """Compute reconstruction R^2, ARI, and AUC for one trained model.

    Zero-shot R^2 is deliberately NOT computed here; see module docstring.
    """
    metrics: dict = {"model": trained.name}

    # Reconstruction R^2 (only if the model reconstructs inputs)
    if trained.reconstruction is not None:
        true = test_features.numpy()
        pred = trained.reconstruction
        for i, var in enumerate(TRAINING_VARIABLES):
            metrics[f"r2_{var}"] = float(r2_score(true[:, i], pred[:, i]))
        metrics["r2_avg"] = float(
            np.mean([metrics[f"r2_{v}"] for v in TRAINING_VARIABLES])
        )
    else:
        metrics["r2_avg"] = float("nan")

    # Clustering ARI
    try:
        clust = evaluate_clustering(trained.embeddings, test_labels)
        metrics["ari"] = float(clust["ari"])
    except Exception as e:
        metrics["ari"] = float("nan")
        metrics["ari_error"] = str(e)

    # Classification AUC
    try:
        cls = evaluate_classification(trained.embeddings, test_labels, top_n=10)
        metrics["auc_macro"] = float(cls["auc_macro"])
    except Exception as e:
        metrics["auc_macro"] = float("nan")
        metrics["auc_error"] = str(e)

    return metrics


# ===========================================================================
# Main bootstrap loop
# ===========================================================================


def run_bootstrap(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    device: str = "auto",
    models: list[str] | None = None,
) -> dict:
    """Run the canonical bootstrap evaluation for training-stability metrics.

    Zero-shot R^2 is not computed here; it should be measured on a
    full-data trained model via `lilyvae-evaluate`.

    Args:
        n_iterations: Number of bootstrap iterations.
        test_fraction: Fraction of boreholes held out per iteration.
        device: "auto", "cpu", "cuda", or "mps".
        models: Optional subset of model names from MODEL_REGISTRY.

    Returns:
        dict with keys: config, per_iteration, summary, elapsed_seconds.
    """
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])

    registry = MODEL_REGISTRY
    if models is not None:
        wanted = set(models)
        registry = [(n, fn) for (n, fn) in MODEL_REGISTRY if n in wanted]
        if not registry:
            raise ValueError(f"No models in MODEL_REGISTRY match {models}")

    unique_boreholes = np.unique(borehole_ids)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))
    rng = np.random.RandomState(SEED)

    per_iteration: list[dict] = []
    start_time = time.time()

    for iteration in range(n_iterations):
        iter_start = time.time()
        print(f"\n{'='*70}")
        print(f"Bootstrap iteration {iteration+1}/{n_iterations}")
        print(f"{'='*70}")

        # Same split seen by every model in this iteration
        perm = rng.permutation(unique_boreholes)
        test_boreholes = set(perm[:n_test])
        test_mask = np.array([bh in test_boreholes for bh in borehole_ids])
        train_mask = ~test_mask

        train_features = features[train_mask]
        train_labels = labels[train_mask]
        test_features = features[test_mask]
        test_labels = labels[test_mask]

        print(
            f"  train: {train_mask.sum()} samples ({len(unique_boreholes)-len(test_boreholes)} boreholes)"
            f" | test: {test_mask.sum()} samples ({len(test_boreholes)} boreholes)"
        )

        iter_result = {
            "iteration": iteration,
            "n_train_samples": int(train_mask.sum()),
            "n_test_samples": int(test_mask.sum()),
            "n_test_boreholes": len(test_boreholes),
            "models": [],
        }

        for model_name, trainer in registry:
            t0 = time.time()
            print(f"\n  [{model_name}]")
            try:
                trained = trainer(train_features, train_labels, test_features, dev)
                metrics = _compute_metrics(trained, test_features, test_labels)
                print(
                    f"    R2={metrics['r2_avg']:.4f}  ARI={metrics['ari']:.4f}  "
                    f"AUC={metrics['auc_macro']:.4f}  ({time.time()-t0:.1f}s)"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                metrics = {"model": model_name, "error": str(e)}
            iter_result["models"].append(metrics)

        iter_result["elapsed_seconds"] = time.time() - iter_start
        per_iteration.append(iter_result)
        print(f"\n  iteration {iteration+1} took {iter_result['elapsed_seconds']:.1f}s")

    elapsed = time.time() - start_time

    summary = _aggregate_summary(per_iteration)

    print(f"\n{'='*70}")
    print(f"Bootstrap complete: {n_iterations} iterations in {elapsed/60:.1f} min")
    print(f"{'='*70}")
    _print_summary(summary)

    return {
        "config": {
            "n_iterations": n_iterations,
            "test_fraction": test_fraction,
            "models": [n for n, _ in registry],
            "seed": SEED,
        },
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }


# ===========================================================================
# Aggregation helpers
# ===========================================================================


def _aggregate_summary(per_iteration: list[dict]) -> dict:
    """Compute median, IQR, 95% CI, mean, std for every (model, metric) pair.

    Also aggregates ``zeroshot_r2_avg`` if the single-point evaluator added
    that key to a model's metrics dict; the key is simply skipped for
    bootstrap outputs that never compute zero-shot.
    """
    metric_keys = [
        "r2_avg", "ari", "auc_macro", "zeroshot_r2_avg",
    ] + [f"r2_{v}" for v in TRAINING_VARIABLES]

    model_names = set()
    for it in per_iteration:
        for m in it["models"]:
            model_names.add(m["model"])

    summary: dict[str, dict] = {}
    for name in sorted(model_names):
        summary[name] = {}
        for key in metric_keys:
            values = []
            for it in per_iteration:
                for m in it["models"]:
                    if m["model"] != name:
                        continue
                    v = m.get(key)
                    if v is None:
                        continue
                    if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
                        continue
                    values.append(float(v))
            if not values:
                continue
            arr = np.array(values)
            summary[name][key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "iqr_low": float(np.percentile(arr, 25)),
                "iqr_high": float(np.percentile(arr, 75)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "n": int(arr.size),
            }
    return summary


def _print_summary(summary: dict) -> None:
    for name, metrics in summary.items():
        print(f"\n  {name}")
        for metric in ("r2_avg", "ari", "auc_macro", "zeroshot_r2_avg"):
            if metric not in metrics:
                continue
            s = metrics[metric]
            print(
                f"    {metric:20s} median={s['median']:.4f} "
                f"IQR=[{s['iqr_low']:.4f}, {s['iqr_high']:.4f}]"
            )
