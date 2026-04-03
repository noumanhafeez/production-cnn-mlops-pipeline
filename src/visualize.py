import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.logger import get_logger


sns.set_theme(style="whitegrid")
logger = get_logger("visualize_result", "logs/visualize_plot.log")

def save_training_plot(history, path="artifacts/training_plot.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Train Loss", marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Train Acc", marker='o')
    if "test_accuracy" in history:
        plt.plot(epochs, history["test_accuracy"], label="Test Acc", marker='x')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Training plot saved at {path}")