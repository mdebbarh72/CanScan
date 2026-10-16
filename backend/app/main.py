
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.schemas import PredictionResponse
from app.predictor import predictTumor, loadModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    loadModel()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="brain tumor classifier",
    description="api that takes a brain MRI image and predicts the tumor class. "
                "For educational use only, not a medical diagnosis.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Brain Tumor API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    imageBytes = await file.read()

    try:
        result = predictTumor(imageBytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image")

    return PredictionResponse(**result)