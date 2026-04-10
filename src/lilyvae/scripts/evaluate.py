"""CLI entry point for evaluating trained LilyVAE models."""

import argparse
import json
import torch
import numpy as np

from lilyvae.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    LATENT_DIM,
    HIDDEN_DIMS,
    INPUT_DIM,
)
from lilyvae.data.dataset import get_dataloader
from lilyvae.model.vae import LilyVAE
from lilyvae.training.trainer import get_device
from lilyvae.evaluation.reconstruction import evaluate_reconstruction
from lilyvae.evaluation.clustering import evaluate_clustering
from lilyvae.evaluation.classification import evaluate_classification
from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot


def load_model(model_type: str, device: torch.device) -> LilyVAE:
    prefix = model_type.replace("-", "_")
    vae = LilyVAE(INPUT_DIM, LATENT_DIM, list(HIDDEN_DIMS)).to(device)
    vae.load_state_dict(torch.load(CHECKPOINT_DIR / f"{prefix}_vae.pt", map_location=device, weights_only=True))
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quick single-run evaluation of trained LilyVAE checkpoints. "
            "NOTE: these numbers are NOT what the paper reports. The paper "
            "uses the canonical bootstrap evaluation (`lilyvae-bootstrap` / "
            "`lilyvae-pipeline`) which trains on 90%% borehole splits and "
            "reports median/IQR across 30 folds. Use this script only for "
            "debugging or sanity-checking a single saved checkpoint."
        )
    )
    parser.add_argument("--model", choices=["unsupervised", "semi-supervised", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    dev = get_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_eval = (
        ["unsupervised", "semi-supervised"] if args.model == "both" else [args.model]
    )

    all_metrics = {}
    dataloader = get_dataloader(batch_size=args.batch_size, shuffle=False)

    # Load labels and metadata
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    for model_type in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type} VAE")
        print(f"{'='*60}")

        prefix = model_type.replace("-", "_")
        vae = load_model(model_type, dev)
        embeddings = torch.load(CHECKPOINT_DIR / f"{prefix}_embeddings.pt", weights_only=True)

        metrics = {}

        # Reconstruction
        print("\nReconstruction R²:")
        recon = evaluate_reconstruction(vae, dataloader, dev)
        for k, v in recon.items():
            print(f"  {k}: {v:.4f}")
            metrics[k] = v

        # Clustering
        print("\nClustering (ARI):")
        clust = evaluate_clustering(embeddings, labels)
        print(f"  ARI: {clust['ari']:.4f}")
        metrics["ari"] = clust["ari"]

        # Classification
        print("\nClassification (AUC):")
        cls_results = evaluate_classification(embeddings, labels)
        print(f"  Macro AUC: {cls_results['auc_macro']:.4f}")
        metrics["auc_macro"] = cls_results["auc_macro"]
        metrics["per_class_auc"] = cls_results["per_class_auc"]

        # Zero-shot
        if not args.skip_zeroshot and zeroshot_targets:
            print("\nZero-shot prediction:")
            zs = evaluate_all_zeroshot(embeddings, zeroshot_targets)
            metrics["zeroshot"] = {k: v for k, v in zs.items()}

        all_metrics[model_type] = metrics

    # Save results
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = RESULTS_DIR / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_metrics), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
