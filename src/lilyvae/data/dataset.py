"""PyTorch Dataset for preprocessed LILY data."""

import torch
from torch.utils.data import Dataset, DataLoader

from lilyvae.config import PROCESSED_DIR, BATCH_SIZE


class LILYDataset(Dataset):
    """Dataset of preprocessed LILY borehole observations."""

    def __init__(self, processed_dir=None):
        processed_dir = processed_dir or PROCESSED_DIR
        self.features = torch.load(processed_dir / "train_features.pt", weights_only=True)
        self.labels = torch.load(processed_dir / "train_labels.pt", weights_only=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


def get_dataloader(
    processed_dir=None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LILYDataset(processed_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
