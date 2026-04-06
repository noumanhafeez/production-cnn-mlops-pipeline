import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

logger = get_logger("data_splitter", "logs/data_splitter.log")


def split_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a DataFrame into training and testing sets for CNN image classification.
    Assumes first column is target labels, remaining columns are pixel values.
    Normalizes pixel values to [0, 1] range.

    Args:
        df (pd.DataFrame): Input dataset with target in first column, pixels in subsequent columns.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) as normalized NumPy arrays.

    Raises:
        ValueError: If DataFrame has fewer than 2 columns.
        Exception: For other unexpected errors.
    """
    try:
        logger.info("Starting data split for CNN training.")

        if df.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns: target + features.")

        # Extract features (pixels) and target (labels)
        X = df.iloc[:, 1:].values  # pixel values
        y = df.iloc[:, 0].values  # labels


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalize pixel values to [0, 1] range (assuming 0-255 input)
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        logger.info(f"Data split completed successfully. "
                    f"Training: X={X_train.shape}, y={y_train.shape}; "
                    f"Testing: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    except ValueError as ve:
        logger.error(f"Data validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data split: {e}")
        raise