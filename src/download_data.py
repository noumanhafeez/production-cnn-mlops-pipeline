import kagglehub
import shutil
import os
from utils.logger import get_logger


logger = get_logger("data_download", "logs/data_download.log")


logger.info("Starting dataset download from KaggleHub...")

# Download dataset
path = kagglehub.dataset_download("zalando-research/fashionmnist")
logger.info(f"Dataset downloaded to cache path: {path}")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
logger.info("Ensured 'data/' directory exists")

# File paths
filename = "fashion-mnist_train.csv"
src_file = os.path.join(path, filename)
dst_file = os.path.join("data", filename)

# Copy file
shutil.copy(src_file, dst_file)
logger.info(f"Dataset copied to project directory: {dst_file}")

logger.info("Dataset preparation completed successfully!")