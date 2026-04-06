import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from utils.logger import get_logger
from pipelines.prediction_pipeline import predict_image

logger = get_logger("api", "logs/api.log")

app = FastAPI(title="Fashion MNIST Prediction API")

MODEL_PATH = "artifacts/model.pth"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the HTML page directly from file
    """
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return HTMLResponse(content=f"<h1>Error loading page: {e}</h1>", status_code=500)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive uploaded file, run prediction, return JSON
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {file.filename}")

        prediction = predict_image(file_path, MODEL_PATH)
        logger.info(f"Prediction: {prediction}")

        os.remove(file_path)

        return JSONResponse({"filename": file.filename, "predicted_class": prediction})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)