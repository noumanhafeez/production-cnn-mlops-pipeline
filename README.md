# Production CNN MLOps Pipeline

A production-ready end-to-end Machine Learning pipeline for image classification using a Convolutional Neural Network (CNN) on the Fashion MNIST dataset. This project follows industry best practices including modular architecture, logging, experiment tracking, CI/CD, and containerization.

---

## Project Overview

This repository implements a complete ML lifecycle:

*  Data ingestion from KaggleHub
*  Data preprocessing and splitting
*  CNN model training with PyTorch
*  Evaluation on test data
*  Experiment tracking using MLflow
*  Model saving and artifact management
*  Prediction pipeline for inference
*  FastAPI-based serving
*  Dockerized deployment
*  CI pipeline with GitHub Actions

---

## Project Structure

```
production-cnn-mlops-pipeline/
│
├── .github/workflows/        # CI pipeline (GitHub Actions)
├── artifacts/                # Saved models & plots
├── config/                   # Configuration files
├── data/                     # Dataset storage
├── frontend/                 # FastAPI application
├── logs/                     # Log files
├── pipelines/                # Training & prediction pipelines
├── src/                      # Core ML modules
├── utils/                    # Utilities (logging, model saving)
│
├── main.py                   # Entry point
├── Dockerfile                # Container configuration
├── requirements.txt          # Dependencies
└── README.md
```

---

##  Demo

![Demo GIF](data/cnn.gif)  


## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/production-cnn-mlops-pipeline.git
cd production-cnn-mlops-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

```bash
python -m src.download_data
```

---

## Train Model

```bash
python main.py
```

Outputs:

* Trained model → `artifacts/model.pth`
* Training plot → `artifacts/training_plot.png`
* Logs → `logs/`

---

## Experiment Tracking (MLflow)

```bash
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

---

## Prediction

Example usage:

```python
from pipelines.prediction_pipeline import predict_image

result = predict_image("image.png", "artifacts/model.pth")
print(result)
```

---

## Run API (FastAPI)

```bash
uvicorn frontend.app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Docker Usage

### Build Image

```bash
docker build -t cnn-mlops .
```

### Run Container

```bash
docker run -p 8000:8000 cnn-mlops
```

---

## CI Pipeline

GitHub Actions automatically:

* Installs dependencies
* Downloads dataset
* Runs training pipeline

File:

```
.github/workflows/ci.yaml
```

---

## Features

* Modular pipeline architecture
* Config-driven training
* Centralized logging system
* MLflow experiment tracking
* Dockerized deployment
* CI/CD integration
* Scalable and production-ready design

---

## Future Improvements

* Add unit and integration tests
* Implement model versioning
* Add data versioning (DVC)
* Deploy to cloud (AWS/GCP/Azure)
* Add monitoring & alerting
* Extend to LLM / RNN pipelines

---

## Author

**Nouman Hafeez**

---

## License

This project is licensed under the MIT License.
