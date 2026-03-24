"""Bootstrap evaluation of VAE training across borehole splits."""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lilyvae.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MASK_FRACTION,
    LATENT_DIM,
    HIDDEN_DIMS,
    INPUT_DIM,
    NUM_LITHOLOGY_CLASSES,
    SEED,
)
from lilyvae.model.vae import LilyVAE
from lilyvae.model.heads import LithologyClassifier
from lilyvae.training.masking import LearnableMasking
from lilyvae.training.loss import (
    heteroscedastic_reconstruction_loss,
    kl_divergence,
    classification_loss,
    beta_schedule,
    alpha_schedule,
)
from lilyvae.training.trainer import get_device
from lilyvae.evaluation.reconstruction import evaluate_reconstruction
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.classification import evaluate_classification


def _train_vae_on_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    model_type: str,
    device: torch.device,
    seed: int,
) -> tuple[LilyVAE, LearnableMasking, LithologyClassifier | None]:
    """Train a VAE on a subset of data. Returns trained model."""
    from lilyvae.config import (
        BETA_START, BETA_END, BETA_ANNEAL_EPOCHS,
        ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS,
    )

    torch.manual_seed(seed)

    dataset = TensorDataset(features, labels, torch.arange(len(features)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    vae = LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)).to(device)
    masking = LearnableMasking(INPUT_DIM, MASK_FRACTION).to(device)

    params = list(vae.parameters()) + list(masking.parameters())
    classifier = None
    if model_type == "semi-supervised":
        classifier = LithologyClassifier(LATENT_DIM, NUM_LITHOLOGY_CLASSES).to(device)
        params += list(classifier.parameters())

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        vae.train()
        masking.train()
        if classifier:
            classifier.train()

        beta = beta_schedule(epoch, BETA_START, BETA_END, BETA_ANNEAL_EPOCHS)
        alpha = alpha_schedule(epoch, ALPHA_START, ALPHA_END, ALPHA_ANNEAL_EPOCHS) if classifier else 0.0

        for batch_features, batch_labels, _ in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            masked = masking(batch_features)
            out = vae(masked)

            recon = heteroscedastic_reconstruction_loss(batch_features, out["x_mu"], out["x_logvar"])
            kl = kl_divergence(out["z_mu"], out["z_logvar"])
            loss = recon + beta * kl

            if classifier:
                logits = classifier(out["z"])
                cls = classification_loss(logits, batch_labels)
                loss = loss + alpha * cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()

    vae.eval()
    masking.eval()
    return vae, masking, classifier


def bootstrap_vae_training(
    model_type: str = "unsupervised",
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    device: str = "auto",
) -> dict:
    """Bootstrap VAE training with borehole-level train/test splits.

    For each iteration:
    1. Split boreholes 90/10
    2. Train VAE from scratch on train boreholes
    3. Evaluate reconstruction R², ARI, AUC on held-out boreholes
    4. Collect held-out predictions for aggregated scatter plot

    Returns dict with per-iteration metrics and aggregated predictions.
    """
    dev = get_device(device)

    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    borehole_ids = metadata["borehole_ids"]

    unique_boreholes = np.unique(borehole_ids)
    n_test = max(1, int(len(unique_boreholes) * test_fraction))
    rng = np.random.RandomState(SEED)

    # Storage for per-iteration metrics
    all_metrics = []
    # Storage for aggregated held-out predictions
    aggregated_true = {}  # sample_idx -> list of true values (should be same)
    aggregated_pred = {}  # sample_idx -> list of predicted values

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Bootstrap iteration {iteration+1}/{n_iterations} ({model_type})")
        print(f"{'='*60}")

        # Split boreholes
        perm = rng.permutation(unique_boreholes)
        test_boreholes = set(perm[:n_test])
        train_boreholes = set(perm[n_test:])

        test_mask = np.array([bh in test_boreholes for bh in borehole_ids])
        train_mask = ~test_mask

        train_features = features[train_mask]
        train_labels = labels[train_mask]
        test_features = features[test_mask]
        test_labels = labels[test_mask]

        print(f"  Train: {train_mask.sum()} samples ({len(train_boreholes)} boreholes)")
        print(f"  Test:  {test_mask.sum()} samples ({len(test_boreholes)} boreholes)")

        # Train VAE
        vae, masking, classifier = _train_vae_on_subset(
            train_features, train_labels, model_type, dev, seed=SEED + iteration,
        )

        # Evaluate on held-out data
        test_dataset = TensorDataset(test_features, test_labels, torch.arange(len(test_features)))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Reconstruction R²
        from sklearn.metrics import r2_score
        all_true = []
        all_pred = []
        all_embeddings = []
        with torch.no_grad():
            for batch_feat, _, _ in test_loader:
                batch_feat = batch_feat.to(dev)
                out = vae(batch_feat)
                all_true.append(batch_feat.cpu().numpy())
                all_pred.append(out["x_mu"].cpu().numpy())
                all_embeddings.append(vae.get_embeddings(batch_feat).cpu())

        true_np = np.concatenate(all_true)
        pred_np = np.concatenate(all_pred)
        embeddings = torch.cat(all_embeddings)

        var_names = ["bulk_density", "magnetic_susceptibility", "NGR", "R", "G", "B"]
        metrics = {"iteration": iteration, "n_test_boreholes": len(test_boreholes), "n_test_samples": int(test_mask.sum())}

        for i, name in enumerate(var_names):
            metrics[f"r2_{name}"] = float(r2_score(true_np[:, i], pred_np[:, i]))
        metrics["r2_avg"] = float(np.mean([v for k, v in metrics.items() if k.startswith("r2_") and k != "r2_avg"]))

        # ARI
        clust = evaluate_clustering(embeddings, test_labels)
        metrics["ari"] = clust["ari"]

        # AUC (common classes only — present in both train and test)
        train_classes = set(train_labels.numpy().tolist())
        test_classes = set(test_labels.numpy().tolist())
        common_classes = train_classes & test_classes
        metrics["n_test_classes"] = len(test_classes)
        metrics["n_common_classes"] = len(common_classes)

        try:
            cls_result = evaluate_classification(embeddings, test_labels, top_n=10)
            metrics["auc_macro"] = cls_result["auc_macro"]
        except Exception as e:
            metrics["auc_macro"] = float("nan")
            print(f"  AUC error: {e}")

        print(f"  R²_avg={metrics['r2_avg']:.4f}  ARI={metrics['ari']:.4f}  AUC={metrics['auc_macro']:.4f}")

        all_metrics.append(metrics)

        # Collect held-out predictions for aggregated plot
        test_indices = np.where(test_mask)[0]
        for j, global_idx in enumerate(test_indices):
            idx = int(global_idx)
            if idx not in aggregated_true:
                aggregated_true[idx] = true_np[j]
                aggregated_pred[idx] = [pred_np[j]]
            else:
                aggregated_pred[idx].append(pred_np[j])

    # Compute summary statistics
    print(f"\n{'='*60}")
    print(f"Bootstrap Summary ({model_type}, {n_iterations} iterations)")
    print(f"{'='*60}")

    metric_keys = ["r2_avg", "r2_bulk_density", "r2_magnetic_susceptibility", "r2_NGR", "r2_R", "r2_G", "r2_B", "ari", "auc_macro"]
    summary = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics if not np.isnan(m.get(key, float("nan")))]
        if values:
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "iqr_low": float(np.percentile(arr, 25)),
                "iqr_high": float(np.percentile(arr, 75)),
                "std": float(np.std(arr)),
            }
            print(f"  {key}: median={summary[key]['median']:.4f} IQR=[{summary[key]['iqr_low']:.4f}, {summary[key]['iqr_high']:.4f}] 95%CI=[{summary[key]['ci_low']:.4f}, {summary[key]['ci_high']:.4f}]")

    # Build aggregated predictions array
    agg_indices = sorted(aggregated_true.keys())
    agg_true = np.array([aggregated_true[i] for i in agg_indices])
    agg_pred = np.array([np.mean(aggregated_pred[i], axis=0) for i in agg_indices])

    return {
        "per_iteration": all_metrics,
        "summary": summary,
        "aggregated_true": agg_true,
        "aggregated_pred": agg_pred,
        "aggregated_indices": np.array(agg_indices),
        "model_type": model_type,
        "n_iterations": n_iterations,
        "test_fraction": test_fraction,
    }
