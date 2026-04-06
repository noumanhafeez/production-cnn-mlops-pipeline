import torch
from torch.utils.data import DataLoader
from utils.logger import get_logger

logger = get_logger("evaluation", "logs/evaluation.log")


def evaluate_model(model, test_loader: DataLoader):
    """
    Evaluate model performance on test dataset.

    Args:
        model (nn.Module): Trained PyTorch model
        test_loader (DataLoader): DataLoader for test data

    Returns:
        float: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            values, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    return accuracy