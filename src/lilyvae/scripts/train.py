"""CLI entry point for training LilyVAE models."""

import argparse
import torch

from lilyvae.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    LATENT_DIM,
    MASK_FRACTION,
    CHECKPOINT_DIR,
)
from lilyvae.data.dataset import get_dataloader
from lilyvae.training.trainer import train_model, save_checkpoint, extract_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train LilyVAE model")
    parser.add_argument(
        "--model",
        choices=["unsupervised", "semi-supervised", "both"],
        default="both",
        help="Which model variant to train",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--mask-fraction", type=float, default=MASK_FRACTION)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    models_to_train = (
        ["unsupervised", "semi-supervised"] if args.model == "both" else [args.model]
    )

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type} VAE")
        print(f"{'='*60}")

        dataloader = get_dataloader(batch_size=args.batch_size)
        result = train_model(
            dataloader=dataloader,
            model_type=model_type,
            num_epochs=args.epochs,
            lr=args.lr,
            mask_fraction=args.mask_fraction,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims,
            device=args.device,
        )

        save_checkpoint(result, model_type)

        # Extract and save embeddings
        eval_loader = get_dataloader(batch_size=args.batch_size, shuffle=False)
        embeddings = extract_embeddings(result["vae"], eval_loader, args.device)
        prefix = model_type.replace("-", "_")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, CHECKPOINT_DIR / f"{prefix}_embeddings.pt")
        print(f"Saved embeddings: {embeddings.shape}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
