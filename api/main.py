
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from Src.model import initialize_model
from Src.predict import predict_image


# Load the trained model when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Railway Defect Detection Model...")

    app.state.model = initialize_model("transfer")

    print("Model loaded successfully!")

    yield

    print("API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Railway Defect Detection API",
    description="Deep Learning based Railway Defect Detection System",
    version="1.0.0",
    lifespan=lifespan
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Allowed image formats
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# Home endpoint
@app.get("/")
async def home():
    return {
        "message": "Railway Defect Detection API is running",
        "status": "success"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    model_loaded = getattr(app.state, "model", None) is not None

    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded
    }


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    temp_path = None

    try:
        image_data = await file.read(MAX_FILE_SIZE + 1)

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(image_data)
            temp_path = temp_file.name

        result = predict_image(
            temp_path,
            app.state.model
        )

        return {
            "status": "success",
            "prediction": result
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()


# API information
@app.get("/api-info")
async def api_info():
    return {
        "project": "Railway Defect Detection",
        "model": "MobileNetV2 Transfer Learning",
        "task": "Image Classification",
        "classes": [
            "Defective",
            "Non defective"
        ],
        "prediction_endpoint": "/predict"
    }
