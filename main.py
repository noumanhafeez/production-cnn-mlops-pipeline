from pipelines.training_pipeline import run_pipeline


if __name__ == "__main__":
    """
    Main script to execute the CNN training pipeline.

    This script serves as the entry point for running the full training workflow:
    - Loads data
    - Prepares DataLoaders
    - Initializes the model
    - Trains the model with logging and MLflow tracking
    - Saves the trained model

    Usage:
        python main.py
    """

    run_pipeline()