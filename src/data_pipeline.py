"""
CNN Data Pipeline Module
Handles dataset creation and batched DataLoader setup for MNIST-like 28x28 image classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from utils.logger import get_logger

logger = get_logger("data_pipeline", "logs/data_pipeline.log")


class CustomDataset(Dataset):
    """
    Custom Dataset for MNIST-style pixel data (28x28 images).

    Args:
        features: NumPy array of shape (N, 784) or flattened pixels
        labels (torch.Tensor): Integer labels of shape (N,)
        img_size (int): Height/width of square images (default=28)
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, img_size: int = 28):
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")

        self.features = features.reshape(-1, 1, img_size, img_size)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size: int = 32, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and testing DataLoaders for CNN training.

    Args:
        X_train, X_test (torch.Tensor): Normalized features [0,1] (N, 784)
        y_train, y_test (torch.Tensor): Integer labels
        batch_size (int): Batch size for training/testing
        num_workers (int): DataLoader workers for parallel loading

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    try:
        logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")

        # Create datasets
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

        logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,  # Faster GPU transfers
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )

        logger.info("DataLoaders created successfully.")
        return train_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}")
        raise