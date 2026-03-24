"""Classification head for semi-supervised VAE variant."""

import torch.nn as nn

from lilyvae.config import LATENT_DIM, NUM_LITHOLOGY_CLASSES


class LithologyClassifier(nn.Module):
    """Linear classification head from latent space to lithology classes."""

    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = NUM_LITHOLOGY_CLASSES):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.linear(z)  # raw logits; softmax applied in cross-entropy loss
