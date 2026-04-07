import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from models.age_predictor import AgePredictor
from utils.image_utils import decode_image

# Configure logging for MLOps tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lightweight Age Prediction API",
    description="MLOps API server for face age prediction using FastAPI and OpenCV",
    version="1.0.0"
)

# Initialize the model instance eager loaded with the server
try:
    predictor = AgePredictor()
    logger.info("AgePredictor initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AgePredictor: {e}")
    predictor = None

@app.get("/")
def health_check():
    """Health check endpoint for orchestration tools (e.g., Kubernetes)."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

@app.post("/predict/age")
async def predict_age(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and receive an age prediction.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    start_time = time.time()
    
    try:
        # 1. Read and decode the image
        file_bytes = await file.read()
        image = decode_image(file_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode the image.")
            
        # 2. Perform prediction via the model wrapper
        prediction = predictor.predict_age(image)
        
        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])
        
        # 3. Return results with pipeline metadata (e.g., latency)
        latency = time.time() - start_time
        logger.info(f"Prediction successful. Latency: {latency:.4f}s")
        
        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "prediction": prediction,
            "meta": {
                "latency_sec": round(latency, 4)
            }
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
