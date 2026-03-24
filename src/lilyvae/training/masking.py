"""Learnable mask tokens for random feature masking."""

import torch
import torch.nn as nn

from lilyvae.config import INPUT_DIM, MASK_FRACTION


class LearnableMasking(nn.Module):
    """Replaces masked features with learnable tokens instead of zeros.

    This lets the model distinguish "this value is missing" from
    "this value equals the mean" (which zero represents after standardization).
    """

    def __init__(self, num_features: int = INPUT_DIM, mask_fraction: float = MASK_FRACTION):
        super().__init__()
        self.mask_fraction = mask_fraction
        self.mask_tokens = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random masking, replacing masked values with learned tokens."""
        if not self.training:
            return x

        mask = torch.rand_like(x) < self.mask_fraction  # True = masked
        mask_values = self.mask_tokens.expand_as(x)
        return torch.where(mask, mask_values, x)
