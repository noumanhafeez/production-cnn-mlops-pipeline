import pandas as pd
from utils.logger import get_logger

logger = get_logger("data_loader", "logs/data_loader.log")

def load_data(path: str) -> pd.DataFrame:
    """
        Load CSV data and log the process.
    """
    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise e