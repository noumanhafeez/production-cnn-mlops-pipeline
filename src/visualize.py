import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")

def plot_training_metrics(history: dict):
    """
    Plot training loss and accuracy over epochs.

    Args:
        history (dict): Dictionary with keys: 'loss', 'accuracy'
                        Each is a list of values per epoch
    """
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()