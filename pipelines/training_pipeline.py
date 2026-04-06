import yaml
import pandas as pd
import torch
import os
from src.model import CNNModel
from src.train import train_model
from utils.model_save import save_model

from src.data_splitter import split_data
from src.data_pipeline import create_data_loaders

from utils.logger import get_logger

logger = get_logger("pipeline", "logs/pipeline.log")


def run_pipeline():
    try:
        logger.info("Pipeline started.")

        config = yaml.safe_load(open("config/config.yaml"))
        logger.info("Config loaded successfully.")

        df = pd.read_csv(config["data"]["path"])
        logger.info(f"Data loaded successfully with shape: {df.shape}")

        X_train, X_test, y_train, y_test = split_data(df)


        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        logger.info("Data converted to PyTorch tensors.")

        train_loader, test_loader = create_data_loaders(
            X_train,
            X_test,
            y_train,
            y_test,
            batch_size=config["training"]["batch_size"]
        )

        model = CNNModel(
            config["model"]["input_channels"],
            config["model"]["num_classes"]
        )

        logger.info("Model initialized successfully.")

        model = train_model(
            model,
            train_loader,
            test_loader,
            config
        )
        logger.info("Model trained successfully.")
        logger.info("Model saved successfully.")

        # Save model
        model_path = config["artifacts"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(model, model_path)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise