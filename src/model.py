import torch
import torch.nn as nn
from utils.logger import get_logger


logger = get_logger("cnn_model", "logs/cnn_model.log")

class CNNModel(nn.Module):
    """
    CNN model for image classification.

    Architecture:
    - Conv2d -> ReLU -> MaxPool2d
    - Conv2d -> ReLU -> MaxPool2d
    - Flatten -> Linear -> ReLU -> Linear (output)

    Args:
        input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        num_classes (int): Number of output classes
    """

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        logger.info(f"Initializing CNNModel with input_channels={input_channels}, num_classes={num_classes}")

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, num_classes)
        )

        logger.info("CNNModel architecture initialized successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Forward pass input shape: {x.shape}")
        x = self.conv(x)
        logger.debug(f"Shape after conv layers: {x.shape}")
        x = self.fc(x)
        logger.debug(f"Forward pass output shape: {x.shape}")
        return x
