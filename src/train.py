import torch
from torch import nn, optim
import mlflow
from utils.logger import get_logger
from evaluate import evaluate_model
from visualize import plot_training_metrics

logger = get_logger("training", "logs/training.log")


def train_model(model, train_loader, test_loader, config):
    """
    Train the CNN model with MLflow tracking, logging, and accuracy.

    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Trained model
    """

    logger.info("Starting training process...")

    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )

        logger.info(f"Optimizer: Adam | Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"Epochs: {config['training']['epochs']}")

        # MLflow setup
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        with mlflow.start_run():
            logger.info("MLflow run started.")
            mlflow.log_params(config["training"])

            history = {"loss": [], "accuracy": [], "test_accuracy": []}

            for epoch in range(config["training"]["epochs"]):
                total_loss = 0.0
                correct = 0
                total = 0

                logger.info(f"Epoch {epoch+1} started.")

                for batch_idx, (X, y) in enumerate(train_loader):
                    logger.debug(f"Batch {batch_idx}: Input shape {X.shape}, Labels shape {y.shape}")

                    outputs = model(X)
                    loss = criterion(outputs, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    values, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)

                avg_loss = total_loss / len(train_loader)
                accuracy = correct / total

                # Append to history
                history["train_loss"].append(avg_loss)
                history["train_accuracy"].append(accuracy)

                # Evaluate on test data
                test_accuracy = evaluate_model(model, test_loader)
                history["test_accuracy"].append(test_accuracy)

                logger.info(
                    f"Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}"
                )

                # Log to MLflow
                mlflow.log_metric("loss", avg_loss, step=epoch)
                mlflow.log_metric("accuracy", accuracy, step=epoch)
                mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

        plot_training_metrics(history)

        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training failed due to error: {str(e)}", exc_info=True)
        raise

    return model